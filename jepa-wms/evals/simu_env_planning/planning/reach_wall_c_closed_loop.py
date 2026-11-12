"""Independent, task-scored Reach-Wall CEM controller with paired A starts.

Run in AI from jepa-wms:
    python -u -X utf8 -m evals.simu_env_planning.planning.reach_wall_c_closed_loop --all

C's ordering: native success first, earlier first hit among successes, smaller
terminal TCP distance among failures. The numeric CEM key only preserves this
ordering; it is not a physical loss or a quantity for mixed-unit regret.
"""

import argparse
import copy
import csv
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import sqlite3
from time import perf_counter

import numpy as np
from omegaconf import OmegaConf
import torch
from tensordict import TensorDict

from evals.simu_env_planning.envs.init import make_env
from evals.simu_env_planning.planning.analyze_candidates import write_csv
from evals.simu_env_planning.planning.common.parser import parse_cfg
from evals.simu_env_planning.planning.reach_wall_b import (
    DEFAULT_RUN, file_hash, restore_decision_state, validate_start_image,
)
from evals.simu_env_planning.planning.reach_wall_b_closed_loop import (
    build_planner, decision_snapshot, global_rng, initial_sampler_state,
    output_lock, raw_actions, restore_global_rng, save_json, save_torch,
)
from evals.simu_env_planning.planning.utils import make_video_mp4
from evals.simu_env_planning.resource_monitor import ResourceMonitor
from evals.utils import make_datasets


def task_rollout(env, bundle, sequence, preprocessor):
    """Real native trajectory, logically stopped at first task success."""
    if env.cfg.task_specification.done_at_succ:
        raise ValueError("Use a separate diagnostic simulator for candidate scoring")
    restore_decision_state(env, bundle)
    actions = raw_actions(sequence, preprocessor)
    if not 0 < len(actions) <= bundle["episode_steps_left"]:
        raise ValueError("C candidate exceeds the decision's remaining raw-action budget")
    low, high = torch.as_tensor(bundle["action_low"]), torch.as_tensor(bundle["action_high"])
    if not (torch.isfinite(actions).all() and (actions >= low-1e-6).all() and (actions <= high+1e-6).all()):
        raise ValueError("C candidate violates the benchmark action box")
    native = env.proprio_env.unwrapped
    distance = float(np.linalg.norm(native.tcp_center - native._target_pos))
    hit = 0 if distance <= .05 else None
    trace = []
    if hit is None:
        for step, action in enumerate(actions, start=1):
            # Preserve native physics and TimeLimit; C does not need candidate images.
            _, _, done, truncated, info = env.env.step(action)
            env._proprios.append(info["proprio"])
            distance, success = float(info["obj_to_target"]), bool(info["success"])
            if not np.isfinite(distance) or distance < 0:
                raise ValueError("Nonfinite native C distance")
            trace.append(dict(distance=distance, success=success))
            if success:
                hit = step
                break
            if done or truncated:
                if step != len(actions):
                    raise ValueError("C simulator terminated before the candidate horizon")
                break
    metrics = dict(task_success=hit is not None, task_first_hit_step=hit,
                   task_steps=hit if hit is not None else len(actions),
                   task_end_distance_m=distance, horizon_raw_actions=len(actions),
                   simulator_actions=len(trace))
    if metrics["task_success"] != (distance <= .05):
        raise ValueError("Native success does not match Reach-Wall's TCP goal threshold")
    return metrics, trace


class TaskObjective:
    """Order-isomorphic numeric key, with no arrival-time/distance trade-off.

    Successful candidates receive -1/(first_hit+1), strictly below every
    failure. Earlier arrival is smaller; equal arrivals tie. Failed candidates
    receive their nonnegative terminal distance. CEM uses costs only to sort
    elites, then updates actions with unweighted elite means/stds. Magnitudes
    never combine steps with metres and must not be used as physical regret.
    """

    def __call__(self, outcomes, actions=None):
        success = outcomes["success"][0].bool()
        hit = outcomes["first_hit_step"][0]
        distance = outcomes["distance_m"][0]
        horizon = outcomes["horizon_raw_actions"][0]
        if (not torch.isfinite(distance).all() or (distance < 0).any()
            or ((hit < 0) & success).any() or ((hit > horizon) & success).any()
            or ((hit != -1) & ~success).any() or (success != (distance <= .05)).any()):
            raise ValueError("Invalid C outcome")
        return torch.where(success, -1. / (hit.clamp_min(0).double() + 1), distance.double())


class OracleTaskUnroll:
    """CEM-compatible physical outcome rollout; no encoder or learned dynamics."""

    def __init__(self, scratch_env, preprocessor, batch_size=32):
        self.env, self.preprocessor, self.batch_size = scratch_env, preprocessor, batch_size
        self.bundle = None
        self.last_rollouts = []
        self.records = {}
        self.simulator_actions = 0

    def set_decision(self, bundle):
        self.bundle = bundle
        self.last_rollouts, self.records, self.simulator_actions = [], {}, 0

    @torch.no_grad()
    def __call__(self, z_init, act_suffix):
        if self.bundle is None or not 0 < act_suffix.shape[1] <= self.batch_size:
            raise ValueError("Set a C decision and respect the configured candidate microbatch size")
        before = global_rng()
        rollouts = []
        try:
            for sequence in act_suffix.detach().cpu().transpose(0, 1):
                metrics, trace = task_rollout(self.env, self.bundle, sequence, self.preprocessor)
                self.records[sequence.contiguous().numpy().tobytes()] = metrics
                self.simulator_actions += len(trace)
                rollouts.append(dict(metrics=metrics, trace=trace))
            self.last_rollouts = rollouts
            metrics = [row["metrics"] for row in rollouts]
            device = act_suffix.device
            return TensorDict(dict(
                success=torch.tensor([[row["task_success"] for row in metrics]], device=device),
                first_hit_step=torch.tensor([[row["task_first_hit_step"] if row["task_first_hit_step"] is not None else -1
                                              for row in metrics]], device=device),
                distance_m=torch.tensor([[row["task_end_distance_m"] for row in metrics]], dtype=torch.float64, device=device),
                horizon_raw_actions=torch.tensor([[row["horizon_raw_actions"] for row in metrics]], device=device),
            ), batch_size=[])
        finally:
            restore_global_rng(before)


def validate_references(scratch, oracle, initial, run, ep, device):
    """Ground-truth C outcomes must reproduce the old fixed-pool measurements."""
    candidates = initial["candidates"][:4]
    oracle.set_decision(initial)
    sequences = torch.stack([row["actions_normalized"] for row in candidates], dim=1).to(device)
    oracle(None, sequences)
    maximum = 0.
    with sqlite3.connect("file:" + (run / "branch_b/scores.sqlite3").as_posix() + "?mode=ro", uri=True) as db:
        for index, rollout in enumerate(oracle.last_rollouts):
            row = db.execute("SELECT payload FROM scores WHERE episode=? AND plan_call=0 AND candidate_index=?", (ep,index)).fetchone()
            if row is None:
                raise ValueError("Missing original fixed-pool native reference")
            old, actual = json.loads(row[0]), rollout["metrics"]
            for name in ("task_success", "task_first_hit_step", "task_steps"):
                if old[name] != actual[name]:
                    raise ValueError(f"C reference disagrees on {name}")
            maximum = max(maximum, abs(old["task_end_distance_m"] - actual["task_end_distance_m"]))
    if maximum > 1e-8:
        raise ValueError("C reference distance differs from the saved native measurement")
    return dict(reference_candidates=4, native_reference_max_distance_error_m=maximum)


def export_summary(run, output, expected):
    completed = []
    for ep in range(expected):
        path = output / f"ep_{ep}" / "episode_result.json"
        if not path.exists():
            continue
        result = json.loads(path.read_text())
        with (run / "mw-reach-wall" / f"ep_{ep}" / "execution.csv").open(newline="") as handle:
            a = list(csv.DictReader(handle))
        b_path = run.parent / "closed_loop_b" / f"ep_{ep}" / "episode_result.json"
        b = json.loads(b_path.read_text()) if b_path.exists() else None
        completed.append(dict(**result, a_success=a[-1]["success"] == "True", a_actions=len(a),
                              b_success=b["success"] if b else None, b_actions=b["actions"] if b else None))
    if completed:
        write_csv(output / "episode_summary.csv", completed)
    summary = dict(expected_episodes=expected, completed_episodes=len(completed), complete=len(completed)==expected,
                   c_successes=sum(row["success"] for row in completed),
                   paired_a_successes=sum(row["a_success"] for row in completed),
                   paired_b_episodes=sum(row["b_success"] is not None for row in completed),
                   paired_b_successes=sum(row["b_success"] is True for row in completed),
                   note="Independent C full episodes; A/B comparisons use these completed C episode IDs only")
    save_json(output / "summary.json", summary)
    return summary


def run_episode(run, output, ep, cfg, actor, scratch, preprocessor, monitor, max_plans=None):
    folder = output / f"ep_{ep}"
    folder.mkdir(exist_ok=True)
    source = run / "mw-reach-wall" / f"ep_{ep}" / "branch_a/plan_000.pt"
    initial = torch.load(source, map_location="cpu", weights_only=False)
    for name in ("action_mean", "action_std"):
        torch.testing.assert_close(getattr(preprocessor,name), initial[name], rtol=0, atol=0)
    device = torch.device("cuda:0")
    oracle = OracleTaskUnroll(scratch, preprocessor, cfg.planner.cost_batch_size)
    objective = TaskObjective()
    monitor.set_context("validate_C_references", episode=ep, plan_call=0)
    checks = validate_references(scratch, oracle, initial, run, ep, device)
    generator = torch.Generator(device=device)
    planner = build_planner(cfg, oracle, preprocessor, generator)
    planner.set_objective(objective)
    checkpoint = folder / "progress.pt"
    if checkpoint.exists():
        progress = torch.load(checkpoint, map_location="cpu", weights_only=False)
        restore_decision_state(actor, progress["decision"])
        generator.set_state(progress["sampler_rng"])
        restore_global_rng(progress["decision"]["rng"])
        visual = torch.stack([torch.as_tensor(frame) for frame in actor._frames])
    else:
        restore_decision_state(actor, initial)
        visual = actor._get_obs()
        checks["actor_start_image_max_error"] = validate_start_image(visual, initial["initial_observation"]["visual"])
        np.testing.assert_array_equal(actor.get_replay_state()["integration_state"], initial["simulator_snapshot"]["integration_state"])
        np.testing.assert_array_equal(actor.proprio_env.unwrapped._target_pos, initial["simulator_goal_xyz"])
        generator.set_state(initial_sampler_state(run, ep, cfg.meta.seed))
        progress = dict(execution=[], plans=[], frames=[visual.clone()], reward=0., success=False, checks=checks)
        progress["decision"] = decision_snapshot(actor, visual, cfg)
        progress["sampler_rng"] = generator.get_state()
        save_torch(checkpoint, progress)
    new_plans = 0
    while actor.steps_left() > 0 and not progress["success"]:
        if max_plans is not None and new_plans >= max_plans:
            return False, new_plans
        call = len(progress["plans"])
        monitor.set_context("planning_C_closed_loop", episode=ep, plan_call=call)
        decision = decision_snapshot(actor, visual, cfg)
        oracle.set_decision(decision)
        steps_left = max((actor.steps_left()+1) // cfg.frameskip, 1)
        print(f"C episode={ep}, call={call}, step={actor.elapsed_steps()}: CEM {cfg.planner.iterations}x{cfg.planner.num_samples}", flush=True)
        started = perf_counter()
        result = planner.plan(torch.zeros(1,device=device), steps_left=steps_left)
        elapsed = perf_counter()-started
        np.testing.assert_array_equal(actor.get_replay_state()["integration_state"], decision["simulator_snapshot"]["integration_state"])
        if actor.elapsed_steps() != decision["episode_steps_elapsed"]:
            raise ValueError("C scoring advanced the acting simulator clock")
        chosen = copy.deepcopy(oracle.last_rollouts[0])
        candidates = []
        for item in result.info["diagnostic_candidates"]:
            saved = {key:value for key,value in item.items() if key not in ("a_cost","predicted_endpoint")}
            saved["c_order_key"] = item["a_cost"]
            saved["actions_simulator"] = raw_actions(saved["actions_normalized"],preprocessor)
            saved.update(oracle.records[saved["actions_normalized"].contiguous().numpy().tobytes()])
            candidates.append(saved)
        if call == 0:
            original = {item["name"]:item for item in initial["candidates"]}
            paired = all(torch.equal(item["actions_normalized"],original[item["name"]]["actions_normalized"])
                         for item in candidates if item["cem_round"]==1 and item["origin"]!="updated_mean")
            if not paired:
                raise ValueError("C initial candidate population is not paired with A")
            progress["checks"]["first_population_matches_a"] = paired
        save_torch(folder/f"plan_{call:03d}.pt",dict(schema_version=1,branch="C_closed_loop",
            source_a_initial_bundle=str(source),episode=ep,plan_call=call,decision=decision,
            native_goal_xyz=initial["simulator_goal_xyz"],candidates=candidates,
            returned_actions_normalized=result.actions.cpu(),final_mean_rollout=chosen))
        monitor.set_context("executing_C",episode=ep,plan_call=call)
        returned = raw_actions(result.actions,preprocessor)
        if not chosen["trace"]:
            raise ValueError("C attempted a new plan from an already successful state")
        for index,action in enumerate(returned):
            visual,reward,done,truncated,info = actor.step(action)
            distance,success = float(info["obj_to_target"]),bool(info["success"])
            expected = chosen["trace"][index]
            if abs(distance-expected["distance"])>1e-8 or success!=expected["success"]:
                raise ValueError("C executed prefix disagrees with its oracle trajectory")
            progress["execution"].append(dict(plan_call=call,simulator_step=actor.elapsed_steps(),
                **{f"action_{d}":float(value) for d,value in enumerate(action)},
                tcp_goal_distance_m=distance,success=success))
            progress["frames"].append(visual.clone())
            progress["reward"] += float(reward)
            progress["success"] = success
            if done or truncated:
                if not success and actor.steps_left()>0:
                    raise ValueError("Unexpected early C episode termination")
                break
        final = next(item for item in candidates if item["is_deployed_output"])
        progress["plans"].append(dict(plan_call=call,start_step=decision["episode_steps_elapsed"],
            executed_steps=actor.elapsed_steps()-decision["episode_steps_elapsed"],
            horizon_raw_actions=final["horizon_raw_actions"],final_mean_order_key=final["c_order_key"],
            final_mean_task_success=final["task_success"],final_mean_first_hit_step=final["task_first_hit_step"],
            final_mean_task_end_distance_m=final["task_end_distance_m"],planning_seconds=elapsed,
            scoring_simulator_actions=oracle.simulator_actions,
            endpoint_tcp_distance_m=progress["execution"][-1]["tcp_goal_distance_m"],success=progress["success"]))
        progress["decision"] = decision_snapshot(actor,visual,cfg)
        progress["sampler_rng"] = generator.get_state()
        save_torch(checkpoint,progress)
        write_csv(folder/"execution.csv",progress["execution"])
        write_csv(folder/"planning_calls.csv",progress["plans"])
        new_plans += 1
        print(f"C episode={ep}, committed call={call}: actions={actor.elapsed_steps()}, "
              f"distance={progress['execution'][-1]['tcp_goal_distance_m']:.5f}m, success={progress['success']}, "
              f"plan_time={elapsed:.1f}s",flush=True)
    monitor.set_context("save_C_episode",episode=ep)
    make_video_mp4(progress["frames"],30,str(folder/"execution.mp4"),obs_concat_channels=False)
    save_json(folder/"episode_result.json",dict(episode=ep,success=progress["success"],actions=actor.elapsed_steps(),
        plan_calls=len(progress["plans"]),end_tcp_goal_distance_m=progress["execution"][-1]["tcp_goal_distance_m"],
        reward=progress["reward"],committed_planning_seconds=sum(item["planning_seconds"] for item in progress["plans"]),
        scoring_simulator_actions=sum(item["scoring_simulator_actions"] for item in progress["plans"])))
    return True,new_plans


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path,default=DEFAULT_RUN)
    parser.add_argument("--output-dir",type=Path)
    selection=parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all",action="store_true")
    selection.add_argument("--episode",type=int)
    parser.add_argument("--max-plans",type=int)
    args=parser.parse_args(argv)
    if args.max_plans is not None and args.max_plans<1:
        parser.error("--max-plans must be positive")
    if not torch.cuda.is_available():
        parser.error("The existing paired CEM sampler requires CUDA")
    logging.disable(logging.INFO)
    run=args.run_dir.resolve()
    output=(args.output_dir or run.parent/"closed_loop_c").resolve()
    if output==run or run in output.parents:
        parser.error("Use an independent output directory outside A's source run")
    output.mkdir(parents=True,exist_ok=True)
    cfg=OmegaConf.load(run/"args_eval.yaml")
    cfg.work_dir,cfg.frameskip=str(output),cfg.model_kwargs.data.custom.frameskip
    cfg=parse_cfg(cfg)
    cfg.planner.cost_batch_size=cfg.planner.get("cost_batch_size",32) or 32
    if (cfg.task_specification.task!="mw-reach-wall" or not cfg.planner.enforce_action_bounds
        or cfg.planner.planner_name!="cem" or not cfg.task_specification.done_at_succ
        or cfg.frameskip!=5 or cfg.model_kwargs.data.custom.action_skip!=1 or cfg.planner.repeat_actskip):
        parser.error("Expected A's bounded Reach-Wall receding-horizon CEM setup")
    expected=cfg.meta.eval_episodes
    if args.episode is not None and not 0<=args.episode<expected:
        parser.error("Episode outside A's captured range")
    selected=range(expected) if args.all else [args.episode]
    protocol=dict(version=1,experiment="independent_C_closed_loop",source_a_run=str(run),
        source_args_sha256=file_hash(run/"args_eval.yaml"),
        initial_bundles={ep:file_hash(run/f"mw-reach-wall/ep_{ep}/branch_a/plan_000.pt") for ep in range(expected)},
        sampler_sources={ep:file_hash(run/f"mw-reach-wall/ep_{ep-1}/episode_result.pt") for ep in range(1,expected)},
        metaworld=importlib.metadata.version("metaworld"),mujoco=importlib.metadata.version("mujoco"),torch=torch.__version__,
        search=OmegaConf.to_container(cfg.planner,resolve=True),
        criterion=dict(success="native TCP distance <= 0.05 m",success_priority="all success outranks failure",
            success_order="earlier first-hit raw action; equal hits tie",failure_order="smaller terminal TCP goal distance",
            cem_key="success: -1/(first_hit+1); failure: terminal distance; order only, no cross-unit physical regret"),
        sampler_pairing="Same initial per-episode CUDA random stream as A/B; C's own updates and states thereafter",
        stopping="Actual execution and candidate task rollouts stop at first native success",
        return_rule="Final elite mean prefix, up to 15 raw actions; 100-action episode budget",
        learned_dynamics=False,encoder_used=False)
    protocol=json.loads(json.dumps(protocol))
    protocol_path=output/"protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text())!=protocol:
        raise ValueError("C output uses different inputs or protocol")
    actor=scratch=None
    with output_lock(output):
        save_json(protocol_path,protocol)
        OmegaConf.save(cfg,output/"args_eval.yaml")
        try:
            with ResourceMonitor(output,interval_s=5) as monitor:
                save_json(output/"status.json",dict(phase="load_normalization",pid=os.getpid()))
                _,preprocessor=make_datasets(cfg.model_kwargs.data,cfg.model_kwargs.data_aug)
                actor=make_env(cfg)
                diagnostic=copy.deepcopy(cfg)
                diagnostic.task_specification.done_at_succ=False
                scratch=make_env(diagnostic)
                used=0
                for ep in selected:
                    if (output/f"ep_{ep}/episode_result.json").exists():
                        print(f"Reuse completed C episode {ep}.",flush=True)
                        continue
                    save_json(output/"status.json",dict(phase="running",episode=ep,pid=os.getpid()))
                    complete,count=run_episode(run,output,ep,cfg,actor,scratch,preprocessor,monitor,
                        max_plans=None if args.max_plans is None else args.max_plans-used)
                    used+=count
                    export_summary(run,output,expected)
                    if not complete or args.max_plans is not None and used>=args.max_plans:
                        break
                summary=export_summary(run,output,expected)
                save_json(output/"status.json",dict(phase="complete" if summary["complete"] else "partial",pid=os.getpid(),**summary))
                print(f"Closed-loop C: {summary['completed_episodes']}/{expected} episodes complete; results: {output}",flush=True)
        except BaseException as exc:
            save_json(output/"status.json",dict(phase="interrupted" if isinstance(exc,KeyboardInterrupt) else "failed",
                episode=locals().get("ep"),error=str(exc),pid=os.getpid()))
            raise
        finally:
            if actor is not None:actor.close()
            if scratch is not None:scratch.close()


if __name__=="__main__":
    main()

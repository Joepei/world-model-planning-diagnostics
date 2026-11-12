"""Independent Reach-Wall B controller: simulator endpoint embeddings inside CEM.

From jepa-wms in conda AI:
    python -X utf8 -m evals.simu_env_planning.planning.reach_wall_b_closed_loop --all

Episode initial physics, goal, and initial sampler RNG are paired with A. All
later planning states and CEM updates belong to B. Completed action prefixes
are checkpointed; repeating a command resumes without repeating committed moves.
"""

import argparse
from contextlib import contextmanager
import copy
import ctypes
from datetime import datetime, timezone
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import random
import sqlite3
from time import perf_counter

import numpy as np
from omegaconf import OmegaConf
import psutil
import torch
from tensordict import TensorDict

from app.vjepa_wm.modelcustom.simu_env_planning.vit_enc_preds import init_module
from evals.simu_env_planning.envs.init import make_env
from evals.simu_env_planning.planning.common.parser import parse_cfg
from evals.simu_env_planning.planning.planning.planner import CEMPlanner
from evals.simu_env_planning.planning.planning.objectives import ReprTargetDistMPCObjective
from evals.simu_env_planning.planning.reach_wall_b import (
    DEFAULT_RUN, REPO, file_hash, restore_decision_state, rollout_candidate,
    validate_pool, validate_start_image,
)
from evals.simu_env_planning.planning.analyze_candidates import write_csv
from evals.simu_env_planning.planning.utils import make_video_mp4
from evals.simu_env_planning.resource_monitor import ResourceMonitor
from evals.utils import make_datasets


REFERENCE_COST_ATOL = 1e-5
REFERENCE_COST_RTOL = 5e-4
GOAL_ENCODING_ATOL = 5e-5


def global_rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch_cpu=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_global_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state.get("cuda"):
        torch.cuda.set_rng_state_all(state["cuda"])


def raw_actions(packed, preprocessor):
    """Same time/frame/component order and denormalization as A's execution."""
    if packed.ndim != 2 or packed.shape[1] != 20:
        raise ValueError("Reach-Wall B requires five packed four-dimensional actions")
    return preprocessor.denormalize_actions(packed.detach().cpu().reshape(-1, 4)).float()


def decision_snapshot(env, visual, cfg):
    snapshot = env.get_replay_state()
    return dict(
        simulator_snapshot=snapshot, initial_state=torch.as_tensor(snapshot["last_state"]).clone(),
        initial_observation=dict(visual=visual.clone(), proprio=env.get_proprios().clone()),
        episode_steps_elapsed=env.elapsed_steps(), episode_steps_left=env.steps_left(),
        pixel_frames=copy.deepcopy(list(env._frames)), proprio_frames=copy.deepcopy(list(env._proprios)),
        action_low=env.action_space.low.copy(), action_high=env.action_space.high.copy(),
        returned_actions_normalized=torch.zeros(min(cfg.planner.num_act_stepped, env.steps_left() // 5), 20),
        rng=global_rng(),
    )


class OracleEndpointUnroll:
    """A CEM-compatible unroll that never calls learned predictive dynamics."""

    def __init__(self, scratch_env, model, preprocessor, batch_size=32):
        self.env, self.model, self.preprocessor = scratch_env, model, preprocessor
        self.batch_size = batch_size
        self.bundle = None
        self.last_rollouts = []

    def set_decision(self, bundle):
        self.bundle = bundle
        self.last_rollouts = []

    @torch.no_grad()
    def __call__(self, z_init, act_suffix):
        if self.bundle is None or not 0 < act_suffix.shape[1] <= self.batch_size:
            raise ValueError("Set a B decision snapshot and respect the encoder microbatch limit")
        before = global_rng()
        endpoints, rollouts = [], []
        try:
            for sequence in act_suffix.detach().cpu().transpose(0, 1):
                candidate = dict(actions_normalized=sequence, actions_simulator=raw_actions(sequence, self.preprocessor))
                endpoint, metrics, trace = rollout_candidate(self.env, self.bundle, candidate)
                endpoints.append(endpoint)
                rollouts.append(dict(metrics=metrics, trace=trace))
            inputs = {key: torch.stack([endpoint[key] for endpoint in endpoints]) for key in ("visual", "proprio")}
            encoded = self.model.encode(inputs)
            result = TensorDict({key: value.transpose(0, 1) for key, value in encoded.items()}, batch_size=[])
            self.last_rollouts = rollouts
            return result
        finally:
            # Scratch replay restores global RNG from a snapshot. It must not
            # perturb real execution or CEM's separate local CUDA generator.
            restore_global_rng(before)


def build_planner(cfg, oracle, preprocessor, generator):
    settings = OmegaConf.to_container(cfg.planner, resolve=True)
    settings.update(action_dim=20, local_generator=generator,
                    action_lower_bounds=preprocessor.normalize_actions(-torch.ones(4)).repeat(5),
                    action_upper_bounds=preprocessor.normalize_actions(torch.ones(4)).repeat(5))
    return CEMPlanner(unroll=oracle, **settings)


def initial_sampler_state(run, ep, seed):
    """Pair the first population with A, then let B's own RNG stream advance."""
    if ep == 0:
        return torch.Generator(device="cuda:0").manual_seed(seed).get_state()
    previous = torch.load(run / "mw-reach-wall" / f"ep_{ep-1}" / "episode_result.pt",
                          map_location="cpu", weights_only=False)
    return previous["rng"]["agent_gpu"]


def save_torch(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def save_json(path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


@contextmanager
def output_lock(output):
    path = output / "active_process.json"
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            old = json.loads(path.read_text())
            try:
                process = psutil.Process(old["pid"])
                if process.create_time() == old["created"]:
                    raise RuntimeError("This closed-loop output directory already has an active runner")
            except psutil.NoSuchProcess:
                pass
            path.unlink()
    with os.fdopen(fd, "w") as handle:
        json.dump(dict(pid=os.getpid(), created=psutil.Process().create_time()), handle)
    awake = bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)) if os.name == "nt" else False
    try:
        yield
    finally:
        if awake:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        path.unlink(missing_ok=True)


def export_summary(run, output, expected):
    completed = []
    for ep in range(expected):
        path = output / f"ep_{ep}" / "episode_result.json"
        if path.exists():
            result = json.loads(path.read_text())
            with (run / "mw-reach-wall" / f"ep_{ep}" / "execution.csv").open(newline="") as handle:
                import csv
                a = list(csv.DictReader(handle))
            completed.append(dict(**result, a_success=a[-1]["success"] == "True", a_actions=len(a)))
    if completed:
        write_csv(output / "episode_summary.csv", completed)
    summary = dict(expected_episodes=expected, completed_episodes=len(completed), complete=len(completed) == expected,
                   paired_a_successes=sum(row["a_success"] for row in completed),
                   b_successes=sum(row["success"] for row in completed),
                   note="Episode outcomes from independently acting B; A comparison includes these completed episode IDs only")
    save_json(output / "summary.json", summary)
    return summary


def validate_reference_costs(actual, expected):
    """Bound cross-run score drift without changing the controller's costs.

    Four reference candidates at all 48 starts reproduced native endpoints
    exactly, with maximum score drift 0.02053%. Rendered observations and GPU
    encoding are not bitwise cross-run measurements. Allow 0.05% plus 1e-5,
    while physics/action/normalization checks remain independent and strict.
    """
    actual, expected = torch.as_tensor(actual).cpu(), torch.as_tensor(expected).cpu()
    if actual.shape != expected.shape or actual.ndim != 1 or not actual.numel():
        raise ValueError("Reference B costs have inconsistent shapes")
    if not (torch.isfinite(actual).all() and torch.isfinite(expected).all()):
        raise ValueError("Reference B costs must be finite")
    differences = (actual - expected).abs()
    allowed = REFERENCE_COST_ATOL + REFERENCE_COST_RTOL * expected.abs()
    if (differences > allowed).any():
        raise ValueError(f"Reference B cost drift exceeds the measured validation bound: "
                         f"actual={actual.tolist()}, saved={expected.tolist()}, "
                         f"absolute_errors={differences.tolist()}, allowed={allowed.tolist()}")
    return float(differences.max())


def validate_reference(oracle, objective, initial, run, ep, device):
    """Reproduce previously measured B scores before deploying the controller."""
    candidates = initial["candidates"][:4]
    oracle.set_decision(initial)
    normalized = torch.stack([candidate["actions_normalized"] for candidate in candidates], dim=1).to(device)
    actual = objective(oracle(None, normalized), normalized).detach().cpu()
    database = run / "branch_b/scores.sqlite3"
    if not database.exists():
        raise ValueError("Expected completed fixed-pool B database for independent reference validation")
    with sqlite3.connect("file:" + database.as_posix() + "?mode=ro", uri=True) as connection:
        references = [json.loads(connection.execute(
            "SELECT payload FROM scores WHERE episode=? AND plan_call=0 AND candidate_index=?", (ep, index)
        ).fetchone()[0]) for index in range(4)]
    native_error = 0.
    for reference, rollout in zip(references, oracle.last_rollouts):
        metric = rollout["metrics"]
        native_error = max(native_error, abs(metric["full_horizon_goal_distance_m"] - reference["full_horizon_goal_distance_m"]))
        if native_error > 1e-8 or metric["task_first_hit_step"] != reference["task_first_hit_step"]:
            raise ValueError("Reference simulator outcomes disagree with the saved B replay")
    expected = [row["b_cost"] for row in references]
    maximum = validate_reference_costs(actual, expected)
    if maximum > 1e-5 + 1e-4 * max(abs(value) for value in expected):
        print(f"B reference validation: max cost drift={maximum:.6g}; native endpoints match; "
              f"bound={REFERENCE_COST_ATOL:g} + {REFERENCE_COST_RTOL:g} * |saved cost|", flush=True)
    return dict(reference_b_cost_max_error=maximum, reference_b_cost_actual=actual.tolist(),
                reference_b_cost_saved=expected, reference_cost_atol=REFERENCE_COST_ATOL,
                reference_cost_rtol=REFERENCE_COST_RTOL, reference_native_distance_max_error=native_error)


def run_episode(run, output, ep, cfg, actor_env, scratch_env, model, preprocessor, monitor, max_plans=None):
    folder = output / f"ep_{ep}"
    folder.mkdir(exist_ok=True)
    source_path = run / "mw-reach-wall" / f"ep_{ep}" / "branch_a/plan_000.pt"
    initial = torch.load(source_path, map_location="cpu", weights_only=False)
    device = torch.device("cuda:0")
    monitor.set_context("validate_B_references", episode=ep, plan_call=0)
    objective, checks = validate_pool(scratch_env, model, initial, device, source_path, goal_atol=GOAL_ENCODING_ATOL)
    checks["goal_encoding_atol"] = GOAL_ENCODING_ATOL
    for name in ("action_mean", "action_std"):
        torch.testing.assert_close(getattr(preprocessor, name), initial[name], rtol=0, atol=0)
    oracle = OracleEndpointUnroll(scratch_env, model, preprocessor, cfg.planner.cost_batch_size)
    checks.update(validate_reference(oracle, objective, initial, run, ep, device))
    save_json(folder / "reference_validation.json", checks)
    generator = torch.Generator(device=device)
    planner = build_planner(cfg, oracle, preprocessor, generator)
    planner.set_objective(objective)
    checkpoint = folder / "progress.pt"
    if checkpoint.exists():
        progress = torch.load(checkpoint, map_location="cpu", weights_only=False)
        restore_decision_state(actor_env, progress["decision"])
        generator.set_state(progress["sampler_rng"])
        restore_global_rng(progress["decision"]["rng"])
        visual = torch.stack([torch.as_tensor(frame) for frame in actor_env._frames])
    else:
        restore_decision_state(actor_env, initial)
        visual = actor_env._get_obs()
        checks["actor_start_image_max_error"] = validate_start_image(visual, initial["initial_observation"]["visual"])
        np.testing.assert_array_equal(actor_env.get_replay_state()["integration_state"], initial["simulator_snapshot"]["integration_state"])
        np.testing.assert_array_equal(actor_env.proprio_env.unwrapped._target_pos, initial["simulator_goal_xyz"])
        generator.set_state(initial_sampler_state(run, ep, cfg.meta.seed))
        progress = dict(execution=[], plans=[], frames=[visual.clone()], reward=0., success=False, checks=checks)
        progress["decision"] = decision_snapshot(actor_env, visual, cfg)
        progress["sampler_rng"] = generator.get_state()
        save_torch(checkpoint, progress)
    new_plans = 0
    while actor_env.steps_left() > 0 and not progress["success"]:
        if max_plans is not None and new_plans >= max_plans:
            return False, new_plans
        call = len(progress["plans"])
        monitor.set_context("planning_B_closed_loop", episode=ep, plan_call=call)
        decision = decision_snapshot(actor_env, visual, cfg)
        oracle.set_decision(decision)
        z = model.encode({key: value.unsqueeze(0) for key, value in decision["initial_observation"].items()})
        # Match PlanEvaluator's horizon shortening and returned final-mean prefix.
        steps_left = max((actor_env.steps_left() + 1) * model.action_skip // cfg.frameskip, 1)
        print(f"B episode={ep}, call={call}, step={actor_env.elapsed_steps()}: CEM {cfg.planner.iterations}x{cfg.planner.num_samples}", flush=True)
        started = perf_counter()
        result = planner.plan(z, steps_left=steps_left)
        elapsed = perf_counter() - started
        # Scoring is performed in a separate simulator and cannot advance actor.
        np.testing.assert_array_equal(actor_env.get_replay_state()["integration_state"], decision["simulator_snapshot"]["integration_state"])
        if actor_env.elapsed_steps() != decision["episode_steps_elapsed"]:
            raise ValueError("Oracle scoring advanced the acting environment clock")
        predicted_trace = oracle.last_rollouts[0]["trace"]
        candidates = []
        for candidate in result.info["diagnostic_candidates"]:
            saved = {key: value for key, value in candidate.items() if key not in ("a_cost", "predicted_endpoint")}
            saved["b_cost"] = candidate["a_cost"]  # CEM's internal field name is generic legacy metadata.
            saved["actions_simulator"] = raw_actions(saved["actions_normalized"], preprocessor)
            candidates.append(saved)
        returned = raw_actions(result.actions, preprocessor)
        if call == 0:
            original = {item["name"]: item for item in initial["candidates"]}
            same = all(torch.equal(item["actions_normalized"], original[item["name"]]["actions_normalized"])
                       for item in candidates if item["cem_round"] == 1 and item["origin"] != "updated_mean")
            progress["checks"]["first_population_matches_a"] = same
            if not same:
                raise ValueError("Paired initial sampler population does not reproduce A")
        # Save B's own candidates and current state; these are not A's later pools.
        save_torch(folder / f"plan_{call:03d}.pt", dict(
            schema_version=1, branch="B_closed_loop", source_a_initial_bundle=str(source_path),
            episode=ep, plan_call=call, decision=decision, goal_observation=initial["goal_observation"],
            goal_encoding=initial["goal_encoding"], latent_objective=initial["latent_objective"],
            candidates=candidates, returned_actions_normalized=result.actions.cpu(),
            final_mean_rollout=oracle.last_rollouts[0]))
        monitor.set_context("executing_B", episode=ep, plan_call=call)
        distance = None
        for index, action in enumerate(returned):
            visual, reward, done, truncated, info = actor_env.step(action)
            distance, success = float(info["obj_to_target"]), bool(info["success"])
            if abs(distance - predicted_trace[index]["distance"]) > 1e-8 or success != predicted_trace[index]["success"]:
                raise ValueError("Executed B prefix disagrees with its oracle rollout")
            progress["execution"].append(dict(plan_call=call, simulator_step=actor_env.elapsed_steps(),
                **{f"action_{d}": float(value) for d, value in enumerate(action)},
                tcp_goal_distance_m=distance, success=success))
            progress["frames"].append(visual.clone())
            progress["reward"] += float(reward)
            progress["success"] = success
            if done or truncated:
                if not success and actor_env.steps_left() > 0:
                    raise ValueError("Unexpected early native termination")
                break
        means = [item for item in candidates if item["origin"] == "updated_mean"]
        progress["plans"].append(dict(plan_call=call, start_step=decision["episode_steps_elapsed"],
            executed_steps=actor_env.elapsed_steps()-decision["episode_steps_elapsed"],
            horizon_raw_actions=len(predicted_trace), final_mean_b_cost=means[-1]["b_cost"],
            last_round_sample_best=float(result.losses[-1]), planning_seconds=elapsed,
            endpoint_tcp_distance_m=distance, success=progress["success"]))
        progress["decision"] = decision_snapshot(actor_env, visual, cfg)
        progress["sampler_rng"] = generator.get_state()
        save_torch(checkpoint, progress)
        write_csv(folder / "execution.csv", progress["execution"])
        write_csv(folder / "planning_calls.csv", progress["plans"])
        new_plans += 1
        print(f"B episode={ep}, committed call={call}: actions={actor_env.elapsed_steps()}, distance={distance:.5f}m, success={progress['success']}, plan_time={elapsed:.1f}s", flush=True)
    monitor.set_context("save_B_episode", episode=ep)
    make_video_mp4(progress["frames"], 30, str(folder / "execution.mp4"), obs_concat_channels=False)
    result = dict(episode=ep, success=progress["success"], actions=actor_env.elapsed_steps(),
                  plan_calls=len(progress["plans"]), end_tcp_goal_distance_m=progress["execution"][-1]["tcp_goal_distance_m"],
                  reward=progress["reward"], committed_planning_seconds=sum(item["planning_seconds"] for item in progress["plans"]))
    save_json(folder / "episode_result.json", result)
    return True, new_plans


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true")
    selection.add_argument("--episode", type=int)
    parser.add_argument("--max-plans", type=int, help="Validate a limited number of new full-budget MPC calls; later commands resume")
    args = parser.parse_args(argv)
    if args.max_plans is not None and args.max_plans < 1:
        parser.error("--max-plans must be positive")
    if not torch.cuda.is_available():
        parser.error("The existing bounded CEM implementation requires CUDA")
    logging.disable(logging.INFO)
    torch.backends.cudnn.benchmark = True
    run = args.run_dir.resolve()
    output = (args.output_dir or run.parent / "closed_loop_b").resolve()
    if output == run or run in output.parents:
        parser.error("Use an independent output directory outside the original A run")
    output.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.load(run / "args_eval.yaml")
    cfg.work_dir, cfg.frameskip = str(output), cfg.model_kwargs.data.custom.frameskip
    cfg = parse_cfg(cfg)
    # A's immutable original args predate its later 32-candidate memory fix.
    # Keep the existing execution safeguard without changing the search budget.
    cfg.planner.cost_batch_size = cfg.planner.get("cost_batch_size", 32) or 32
    if (cfg.task_specification.task != "mw-reach-wall" or not cfg.planner.enforce_action_bounds
        or cfg.planner.planner_name != "cem" or not cfg.task_specification.done_at_succ
        or cfg.frameskip != 5 or cfg.model_kwargs.data.custom.action_skip != 1 or cfg.planner.repeat_actskip
        or cfg.planner.planning_objective.sum_all_diffs):
        parser.error("Expected A's bounded, stop-on-success Reach-Wall endpoint-CEM protocol")
    expected = cfg.meta.eval_episodes
    if args.episode is not None and not 0 <= args.episode < expected:
        parser.error("Episode outside A's captured range")
    selected = range(expected) if args.all else [args.episode]
    settings = OmegaConf.to_container(cfg.model_kwargs, resolve=True)
    checkpoint = Path(settings["checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = (REPO / cfg.get("checkpoint_folder", ".") / checkpoint).resolve()
    protocol = dict(version=1, experiment="independent_B_closed_loop", source_a_run=str(run),
        source_args_sha256=file_hash(run / "args_eval.yaml"), checkpoint_sha256=file_hash(checkpoint),
        metaworld=importlib.metadata.version("metaworld"), mujoco=importlib.metadata.version("mujoco"), torch=torch.__version__,
        initial_bundles={ep: file_hash(run / "mw-reach-wall" / f"ep_{ep}" / "branch_a/plan_000.pt") for ep in range(expected)},
        sampler_sources={ep: file_hash(run / "mw-reach-wall" / f"ep_{ep-1}" / "episode_result.pt") for ep in range(1, expected)},
        sampler_pairing="Each trial starts at A's initial CUDA sampler RNG; subsequent CEM sampling belongs to B",
        search=OmegaConf.to_container(cfg.planner, resolve=True),
        actor="Restored A episode initial physics/goal; thereafter B acts and observes its own environment",
        scoring="Scratch simulator full candidate horizon -> same actual-image/proprio encoder -> same endpoint latent objective",
        return_rule="Final elite mean, first 3 model steps (up to 15 raw actions), then replan; native success stops immediately")
    protocol = json.loads(json.dumps(protocol))
    protocol_path = output / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Existing closed-loop output has a different protocol or changed inputs")
    status_path = output / "status.json"
    actor_env = scratch_env = None
    with output_lock(output):
        save_json(protocol_path, protocol)
        save_json(output / "validation_policy.json", dict(
            revision=2, reference_cost_atol=REFERENCE_COST_ATOL, reference_cost_rtol=REFERENCE_COST_RTOL,
            goal_encoding_atol=GOAL_ENCODING_ATOL, goal_encoding_rtol=1e-4,
            native_reference_distance_atol=1e-8,
            note="Validation calibration only; CEM scores, simulation, objective, and completed results are unchanged"))
        OmegaConf.save(cfg, output / "args_eval.yaml")
        try:
            with ResourceMonitor(output, interval_s=5) as monitor:
                save_json(status_path, dict(phase="load_encoder", pid=os.getpid(), requested_episodes=list(selected)))
                monitor.set_context("load_encoder")
                dset, preprocessor = make_datasets(cfg.model_kwargs.data, cfg.model_kwargs.data_aug)
                model = init_module(folder=str(checkpoint.parent), checkpoint=checkpoint.name,
                    model_kwargs=settings["pretrain_kwargs"], device=torch.device("cuda:0"),
                    action_dim=dset.action_dim, proprio_dim=dset.proprio_dim, preprocessor=preprocessor,
                    cfgs_data=settings["data"], wrapper_kwargs=settings.get("wrapper_kwargs", {}))
                model.eval()
                actor_env = make_env(cfg)
                scratch_cfg = copy.deepcopy(cfg)
                scratch_cfg.task_specification.done_at_succ = False
                scratch_env = make_env(scratch_cfg)
                used = 0
                for ep in selected:
                    if (output / f"ep_{ep}" / "episode_result.json").exists():
                        print(f"Reuse completed B episode {ep}.", flush=True)
                        continue
                    save_json(status_path, dict(phase="running", episode=ep, pid=os.getpid()))
                    complete, count = run_episode(run, output, ep, cfg, actor_env, scratch_env, model, preprocessor, monitor,
                        max_plans=None if args.max_plans is None else args.max_plans-used)
                    used += count
                    summary = export_summary(run, output, expected)
                    if not complete or args.max_plans is not None and used >= args.max_plans:
                        break
                summary = export_summary(run, output, expected)
                save_json(status_path, dict(phase="complete" if summary["complete"] else "partial", **summary, pid=os.getpid()))
                print(f"Closed-loop B: {summary['completed_episodes']}/{expected} episodes complete; results: {output}", flush=True)
        except BaseException as exc:
            save_json(status_path, dict(phase="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                episode=locals().get("ep"), error=str(exc), pid=os.getpid()))
            raise
        finally:
            if actor_env is not None:
                actor_env.close()
            if scratch_env is not None:
                scratch_env.close()


if __name__ == "__main__":
    main()

"""Rescore A's fixed candidate pools with simulator dynamics and latent cost.

This is a decision-local experiment at A's visited states, not a deployed B
controller. Task first-hit diagnostics stop logically at success; B's endpoint
cost uses counterfactual continuation over the entire saved candidate horizon.
"""

import argparse
import csv
import hashlib
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
import torch
from tensordict import TensorDict

from app.vjepa_wm.modelcustom.simu_env_planning.vit_enc_preds import init_module
from evals.simu_env_planning.envs.init import make_env
from evals.simu_env_planning.planning.common.parser import parse_cfg
from evals.simu_env_planning.planning.planning.objectives import ReprTargetDistMPCObjective
from evals.simu_env_planning.resource_monitor import ResourceMonitor
from evals.utils import make_datasets

PROJECT = Path(__file__).resolve().parents[4]
REPO = PROJECT / "jepa-wms"
DEFAULT_RUN = PROJECT / "runs/reach_wall/simu_env_planning/bounded_a_i15_n300_ep48"
PROTOCOL_VERSION = 1
METRIC_KEYS = (
    "b_cost", "full_horizon_steps", "full_horizon_goal_distance_m", "full_horizon_success",
    "task_success", "task_first_hit_step", "task_steps", "task_end_distance_m",
    "prefix_steps", "prefix_success", "prefix_horizon_goal_distance_m", "prefix_task_steps", "prefix_task_end_distance_m",
)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def route_id(candidate):
    return hashlib.sha256(candidate["actions_normalized"].contiguous().numpy().tobytes()).hexdigest()


def discover_bundles(run_dir):
    paths = [path for path in Path(run_dir).glob("mw-reach-wall/ep_*/branch_a/plan_*.pt")
             if path.parents[1].name[3:].isdigit()]
    return sorted(paths, key=lambda path: (int(path.parents[1].name[3:]), int(path.stem[5:])))


def restore_decision_state(env, bundle):
    """Restore every candidate to the same physics, clock, history, and RNG."""
    restored = env.set_replay_state(bundle["simulator_snapshot"])
    np.testing.assert_allclose(restored, torch.as_tensor(bundle["initial_state"]).numpy(), atol=1e-7, rtol=0)
    env.set_elapsed_steps(bundle["episode_steps_elapsed"])
    for buffer, key in ((env._frames, "pixel_frames"), (env._proprios, "proprio_frames")):
        buffer.clear()
        buffer.extend(value.clone() if isinstance(value, torch.Tensor) else value.copy() for value in bundle[key])
    random.setstate(bundle["rng"]["python"])
    np.random.set_state(bundle["rng"]["numpy"])
    torch.set_rng_state(bundle["rng"]["torch_cpu"])
    torch.testing.assert_close(env.get_proprios(), bundle["initial_observation"]["proprio"], atol=1e-7, rtol=0)


def rollout_candidate(env, bundle, candidate, render_each_step=False):
    """Simulate all commands; record task stopping and prefix outcomes separately.

    This task's encoder uses one image and one proprioception frame, so only the
    final image needs rendering. The full rendering path is retained for checks.
    """
    if env.num_frames != 1 or env.min_proprios != 1:
        raise ValueError("Endpoint-only replay currently requires one image and one proprioception frame")
    if env.cfg.task_specification.done_at_succ:
        raise ValueError("B diagnostic replay must allow full-horizon continuation after native success")
    restore_decision_state(env, bundle)
    actions = candidate["actions_simulator"].cpu()
    if actions.ndim != 2 or actions.shape[1] != 4 or not 0 < len(actions) <= bundle["episode_steps_left"]:
        raise ValueError("Candidate shape/horizon exceeds the saved decision's remaining action budget")
    if not (torch.isfinite(actions).all() and (actions >= torch.as_tensor(bundle["action_low"]) - 1e-6).all()
            and (actions <= torch.as_tensor(bundle["action_high"]) + 1e-6).all()):
        raise ValueError("Candidate commands violate the benchmark action bounds")
    prefix_steps = min(len(actions), len(bundle["returned_actions_normalized"]) * (actions.shape[0] // candidate["actions_normalized"].shape[0]))
    native = env.proprio_env.unwrapped
    initial_distance = float(np.linalg.norm(native.tcp_center - native._target_pos))
    first_hit = 0 if initial_distance <= .05 else None
    task_end_distance = initial_distance if first_hit == 0 else None
    trace = []
    prefix_distance = None
    for step, action in enumerate(actions, start=1):
        if render_each_step:
            visual, _, done, truncated, info = env.step(action)
        else:
            # Preserve native dynamics/TimeLimit/TensorWrapper, skip intermediate
            # PixelWrapper rendering, and keep its proprioception buffer current.
            _, _, done, truncated, info = env.env.step(action)
            env._proprios.append(info["proprio"])
        distance, success = float(info["obj_to_target"]), bool(info["success"])
        if not np.isfinite(distance):
            raise ValueError("Nonfinite native task distance")
        trace.append(dict(distance=distance, success=success))
        if success and first_hit is None:
            first_hit, task_end_distance = step, distance
        if step == prefix_steps:
            prefix_distance = distance
        if (done or truncated) and step < len(actions):
            raise ValueError("Simulator terminated before the complete candidate horizon")
    if not render_each_step:
        visual = env._get_obs()
    endpoint = {"visual": visual.clone(), "proprio": env.get_proprios().clone()}
    metrics = dict(
        full_horizon_steps=len(actions), full_horizon_goal_distance_m=distance, full_horizon_success=success,
        task_success=first_hit is not None, task_first_hit_step=first_hit,
        task_steps=first_hit if first_hit is not None else len(actions),
        task_end_distance_m=task_end_distance if first_hit is not None else distance,
        prefix_steps=prefix_steps, prefix_success=first_hit is not None and first_hit <= prefix_steps,
        prefix_horizon_goal_distance_m=prefix_distance,
        prefix_task_steps=first_hit if first_hit is not None and first_hit <= prefix_steps else prefix_steps,
        prefix_task_end_distance_m=task_end_distance if first_hit is not None and first_hit <= prefix_steps else prefix_distance,
    )
    return endpoint, metrics, trace


@torch.inference_mode()
def encode_actual_endpoints(model, objective, endpoints):
    """Image/proprioception -> original encoder -> original endpoint objective."""
    inputs = {key: torch.stack([endpoint[key] for endpoint in endpoints]) for key in ("visual", "proprio")}
    encoded = model.encode(inputs)
    # Encoder: [batch, time, ...]; objective: [time, batch, ...].
    encoded = TensorDict({key: value.transpose(0, 1) for key, value in encoded.items()}, batch_size=[])
    costs = objective(encoded, None).detach().cpu()
    if costs.shape != (len(endpoints),) or not torch.isfinite(costs).all():
        raise ValueError("B must return one finite latent cost per endpoint")
    return costs.tolist()


def validate_start_image(start, saved):
    """Allow sparse OpenGL rasterization outliers, reject a changed scene.

    Physics is independently checked against the saved state and every recorded
    deployed-prefix distance/success. The image is not expected to be bit exact.
    """
    if start.shape != saved.shape:
        raise ValueError("Restored image shape differs from A")
    difference = (start.float() - saved.float()).abs()
    maximum = difference.max().item()
    mean = difference.mean().item()
    sparse_fraction = (difference > 1).float().mean().item()
    if mean > .01 or sparse_fraction > 1e-4:
        raise ValueError(f"Restored scene differs from A: max={maximum}, mean={mean}, fraction_above_1={sparse_fraction}")
    if maximum > 1:
        print(f"Sparse render difference accepted: max={maximum:.0f}, mean={mean:.6g}, fraction_above_1={sparse_fraction:.6g}; native prefix still checked strictly", flush=True)
    return maximum


def validate_pool(env, model, bundle, objective_device, bundle_path, *, goal_atol=1e-5):
    """Check start images, goal encoding, and A's actually executed prefix."""
    if bundle["schema_version"] != 2 or bundle["task"] != "mw-reach-wall":
        raise ValueError("Expected a Reach-Wall simulator snapshot")
    if bundle["latent_objective"]["type"] != "L2" or bundle["latent_objective"]["sum_all_diffs"]:
        raise ValueError("This replay implements the saved endpoint L2 objective only")
    restore_decision_state(env, bundle)
    start = env._get_obs().float()
    saved = bundle["initial_observation"]["visual"].float()
    image_error = validate_start_image(start, saved)
    with torch.inference_mode():
        checked = model.encode({key: value.unsqueeze(0) for key, value in bundle["goal_observation"].items()})
    for key in ("visual", "proprio"):
        torch.testing.assert_close(checked[key].cpu(), bundle["goal_encoding"][key], atol=goal_atol, rtol=1e-4)
    goal = TensorDict({key: value.to(objective_device) for key, value in bundle["goal_encoding"].items()}, batch_size=[])
    objective = ReprTargetDistMPCObjective(env.cfg, goal, **{
        key: bundle["latent_objective"][key] for key in ("alpha", "sum_all_diffs")
    })
    deployed = [candidate for candidate in bundle["candidates"] if candidate["is_deployed_output"]]
    if len(deployed) != 1:
        raise ValueError("Expected one deployed final mean per planning call")
    _, _, trace = rollout_candidate(env, bundle, deployed[0])
    with (bundle_path.parents[1] / "execution.csv").open(newline="", encoding="utf-8") as handle:
        recorded = [row for row in csv.DictReader(handle) if int(row["plan_call"]) == bundle["plan_call"]]
    if not recorded or len(recorded) > len(trace):
        raise ValueError("Missing deployed execution records")
    for index, row in enumerate(recorded):
        torch.testing.assert_close(deployed[0]["actions_simulator"][index],
                                   torch.tensor([float(row[f"action_{d}"]) for d in range(4)]), atol=1e-6, rtol=0)
        if abs(trace[index]["distance"] - float(row["tcp_goal_distance_m"])) > 1e-8:
            raise ValueError("Restored simulator does not reproduce A's executed native distances")
        if trace[index]["success"] != (row["success"] == "True"):
            raise ValueError("Restored simulator does not reproduce A's native success")
    return objective, dict(start_image_max_error=image_error, deployed_prefix_verified_steps=len(recorded))


class ScoreStore:
    """A transaction commits each scoring batch, preserving work across restarts."""

    def __init__(self, path, protocol):
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS pools (episode INTEGER, plan_call INTEGER, source_sha256 TEXT,
                total INTEGER, PRIMARY KEY (episode, plan_call));
            CREATE TABLE IF NOT EXISTS scores (episode INTEGER, plan_call INTEGER, candidate_index INTEGER,
                payload TEXT NOT NULL, PRIMARY KEY (episode, plan_call, candidate_index));
        """)
        encoded = json.dumps(protocol, sort_keys=True)
        existing = self.connection.execute("SELECT value FROM metadata WHERE key='protocol'").fetchone()
        if existing and existing[0] != encoded:
            self.connection.close()
            raise ValueError("Existing B database uses different inputs/protocol; use another --output-dir")
        self.connection.execute("INSERT OR IGNORE INTO metadata VALUES ('protocol', ?)", (encoded,))
        self.connection.commit()

    def ensure_pool(self, episode, plan_call, source_sha, total):
        old = self.connection.execute("SELECT source_sha256,total FROM pools WHERE episode=? AND plan_call=?", (episode, plan_call)).fetchone()
        if old and old != (source_sha, total):
            raise ValueError("An A candidate bundle changed after B scoring began")
        with self.connection:
            self.connection.execute("INSERT OR IGNORE INTO pools VALUES (?,?,?,?)", (episode, plan_call, source_sha, total))

    def rows(self, episode, plan_call):
        return {index: json.loads(payload) for index, payload in self.connection.execute(
            "SELECT candidate_index,payload FROM scores WHERE episode=? AND plan_call=? ORDER BY candidate_index", (episode, plan_call))}

    def commit(self, rows):
        with self.connection:
            self.connection.executemany("INSERT INTO scores VALUES (?,?,?,?)", [
                (row["episode"], row["plan_call"], row["candidate_index"], json.dumps(row, allow_nan=False)) for row in rows
            ])

    def export(self, path, pool=None):
        query = "SELECT payload FROM scores"
        parameters = ()
        if pool is not None:
            query += " WHERE episode=? AND plan_call=?"
            parameters = pool
        query += " ORDER BY episode,plan_call,candidate_index"
        cursor = self.connection.execute(query, parameters)
        first = cursor.fetchone()
        if first is None:
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".csv.tmp")
        with temp.open("w", newline="", encoding="utf-8") as handle:
            row = json.loads(first[0])
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
            for (payload,) in cursor:
                writer.writerow(json.loads(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)

    def export_selections(self, path):
        """Report choices only for fully scored pools; no C ranking is implied."""
        selections = []
        for ep, call, total in self.connection.execute("SELECT episode,plan_call,total FROM pools ORDER BY episode,plan_call"):
            rows = list(self.rows(ep,call).values())
            if len(rows) != total:
                continue
            deployed = [row for row in rows if row["is_deployed_output"]]
            if len(deployed) != 1:
                raise ValueError("A completed pool must contain exactly one deployed final mean")
            for name, row in (("a_deployed_mean",deployed[0]),
                              ("a_pool_min",min(rows,key=lambda row:row["a_cost"])),
                              ("b_pool_min",min(rows,key=lambda row:row["b_cost"]))):
                selections.append(dict(selection=name,pool_occurrences=total,**row))
        if not selections:
            return
        path = Path(path)
        temp = path.with_suffix(".csv.tmp")
        with temp.open("w",newline="",encoding="utf-8") as handle:
            writer = csv.DictWriter(handle,fieldnames=list(selections[0]))
            writer.writeheader()
            writer.writerows(selections)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp,path)


def score_pool(path, cfg, env, model, device, store, output, batch_size, limit, monitor):
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    ep, call = bundle["episode"], bundle["plan_call"]
    candidates = bundle["candidates"]
    store.ensure_pool(ep, call, file_hash(path), len(candidates))
    existing = store.rows(ep, call)
    maximum = min(limit, len(candidates)) if limit else len(candidates)
    missing = [index for index in range(maximum) if index not in existing]
    if missing:
        monitor.set_context("validate_replay", episode=ep, plan_call=call)
        objective, checks = validate_pool(env, model, bundle, device, path)
        # An identical route is replayed once within this decision state. Keep
        # every occurrence's original A score and CEM metadata in the output.
        cache = {row["route_id"]: {key: row[key] for key in METRIC_KEYS} for row in existing.values()}
        for start in range(0, len(missing), batch_size):
            indices = missing[start : start + batch_size]
            pending, endpoints = {}, []
            monitor.set_context("simulate_candidates", episode=ep, plan_call=call)
            for index in indices:
                candidate = candidates[index]
                key = route_id(candidate)
                if key not in cache and key not in pending:
                    endpoint, metrics, _ = rollout_candidate(env, bundle, candidate)
                    pending[key] = metrics
                    endpoints.append(endpoint)
            if endpoints:
                monitor.set_context("encode_endpoints", episode=ep, plan_call=call)
                costs = encode_actual_endpoints(model, objective, endpoints)
                for (key, metrics), cost in zip(pending.items(), costs):
                    cache[key] = dict(metrics, b_cost=cost)
            rows = []
            for index in indices:
                candidate = candidates[index]
                key = route_id(candidate)
                rows.append(dict(
                    task="mw-reach-wall", episode=ep, plan_call=call, candidate_index=index,
                    candidate=candidate["name"], cem_round=candidate["cem_round"],
                    origin=candidate["origin"], sample_index=candidate["sample_index"], is_elite=candidate["is_elite"],
                    is_deployed_output=candidate["is_deployed_output"], is_executed_prefix=candidate["is_executed_prefix"],
                    route_id=key, a_cost=float(candidate["a_cost"]),
                    horizon_model_steps=candidate["actions_normalized"].shape[0],
                    episode_steps_elapsed=bundle["episode_steps_elapsed"], episode_steps_left=bundle["episode_steps_left"],
                    encoder_batch_size=batch_size, **{name: cache[key][name] for name in METRIC_KEYS}, **checks,
                ))
            store.commit(rows)
            completed = len(existing) + min(start + batch_size, len(missing))
            if start == 0 or completed % 256 < batch_size or start + batch_size >= len(missing):
                print(f"episode={ep} plan_call={call}: {completed}/{len(candidates)} occurrences saved", flush=True)
    else:
        print(f"episode={ep} plan_call={call}: reuse {len(existing)}/{len(candidates)} saved scores", flush=True)
    store.export(output / "mw-reach-wall" / f"ep_{ep}" / f"plan_{call:03d}.csv", (ep, call))
    return len(store.rows(ep, call)), len(candidates)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--episode", type=int, help="Score this episode's decision pools")
    selection.add_argument("--all", action="store_true", help="Score all saved A decision pools")
    parser.add_argument("--plan-call", type=int, help="Select one call within --episode")
    parser.add_argument("--max-candidates", type=int, help="Small initial check; later invocations fill remaining rows")
    parser.add_argument("--encoder-batch-size", type=int, default=32)
    args = parser.parse_args(argv)
    if args.plan_call is not None and args.episode is None:
        parser.error("--plan-call requires --episode")
    if args.encoder_batch_size < 1 or (args.max_candidates is not None and args.max_candidates < 1):
        parser.error("Batch size and candidate limit must be positive")
    logging.disable(logging.INFO)
    torch.backends.cudnn.benchmark = True
    run = args.run_dir.resolve()
    output = (args.output_dir or run / "branch_b").resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = [path for path in discover_bundles(run)
             if (args.episode is None or int(path.parents[1].name[3:]) == args.episode)
             and (args.plan_call is None or int(path.stem[5:]) == args.plan_call)]
    if not paths:
        parser.error("No matching A candidate bundles")
    cfg = OmegaConf.load(run / "args_eval.yaml")
    cfg.work_dir, cfg.frameskip = str(run), cfg.model_kwargs.data.custom.frameskip
    cfg = parse_cfg(cfg)
    if cfg.task_specification.task != "mw-reach-wall" or not cfg.planner.enforce_action_bounds:
        parser.error("Expected the bounded Reach-Wall A evaluation")
    # Only the diagnostic simulator continues after task success. Source A args,
    # task first-hit measurements, and the original controller stay unchanged.
    cfg.task_specification.done_at_succ = False
    settings = OmegaConf.to_container(cfg.model_kwargs, resolve=True)
    checkpoint = Path(settings["checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = (REPO / cfg.get("checkpoint_folder", ".") / checkpoint).resolve()
    protocol = dict(version=PROTOCOL_VERSION, original_args_sha256=file_hash(run / "args_eval.yaml"),
                    checkpoint_sha256=file_hash(checkpoint), torch_version=torch.__version__,
                    metaworld_version=importlib.metadata.version("metaworld"), mujoco_version=importlib.metadata.version("mujoco"),
                    b_endpoint="full_saved_candidate_horizon", task_stopping="first_native_success_le_0.05m",
                    experiment="fixed_A_pools_at_A_visited_states", encoder_batch_size_note="recorded per row; may vary on resume")
    store = ScoreStore(output / "scores.sqlite3", protocol)
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    env = None
    start = perf_counter()
    try:
        with ResourceMonitor(output, interval_s=5) as monitor:
            monitor.set_context("load_encoder")
            dset, preprocessor = make_datasets(cfg.model_kwargs.data, cfg.model_kwargs.data_aug)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model = init_module(folder=str(checkpoint.parent), checkpoint=checkpoint.name,
                                model_kwargs=settings["pretrain_kwargs"], device=device,
                                action_dim=dset.action_dim, proprio_dim=dset.proprio_dim, preprocessor=preprocessor,
                                cfgs_data=settings["data"], wrapper_kwargs=settings.get("wrapper_kwargs", {}))
            model.eval()
            env = make_env(cfg)
            print(f"B replay: {len(paths)} selected pools; endpoint encoding batch <= {args.encoder_batch_size}", flush=True)
            completed_pools = 0
            for path in paths:
                count, total = score_pool(path, cfg, env, model, device, store, output,
                                          args.encoder_batch_size, args.max_candidates, monitor)
                completed_pools += count == total
            store.export(output / "scored_candidates.csv")
            store.export_selections(output / "pool_selections.csv")
            print(f"Done: {completed_pools}/{len(paths)} selected pools complete; {perf_counter()-start:.1f}s elapsed.\nCSV: {output / 'scored_candidates.csv'}", flush=True)
    finally:
        if env is not None:
            env.close()
        store.connection.close()


if __name__ == "__main__":
    main()

"""Episode-boundary checkpoints for the single-process Reach-Wall evaluation.

Old completed trials can be recovered from their recorded actions and simulator
snapshots without running the world model. Incomplete trials restart at their
original episode index; their partial files are archived first.
"""

import copy
import csv
import hashlib
import json
import logging
import os
from pathlib import Path
import random
from datetime import datetime, timezone

import numpy as np
import torch
from omegaconf import OmegaConf

log = logging.getLogger(__name__)
CHECKPOINT = "episode_result.pt"


def config_digest(config):
    config = OmegaConf.to_container(config, resolve=True) if OmegaConf.is_config(config) else copy.deepcopy(config)
    config.get("meta", {}).pop("resume_completed", None)
    # Execution-only changes are recorded separately and can be adjusted on
    # resume. All scientific parameters (population, rounds, seeds, etc.) remain
    # protected. Removing absent fields also preserves existing checkpoint hashes.
    config.get("planner", {}).pop("cost_batch_size", None)
    config.get("logging", {}).pop("resource_monitor_interval_s", None)
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def record_execution_session(folder, config):
    """Keep the actual execution config without overwriting original run args."""
    record = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(), "config": config}
    with (Path(folder) / "execution_sessions.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def check_resume_config(current, previous):
    if config_digest(current) != config_digest(previous):
        raise ValueError("Resume config differs from saved args_eval.yaml. Restore the original settings before resuming.")


def _rng_state(agent):
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "agent_cpu": agent.local_generator.get_state(),
        "agent_gpu": agent.local_gpu_generator.get_state(),
    }


def _restore_rng(state, agent):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    agent.local_generator.set_state(state["agent_cpu"])
    agent.local_gpu_generator.set_state(state["agent_gpu"])


def save_episode_checkpoint(path, digest, agent, env, values, elapsed, source="evaluation"):
    payload = {
        "schema_version": 1,
        "config_digest": digest,
        "values": tuple(float(value) for value in values),
        "elapsed": float(elapsed),
        "source": source,
        "cost_batch_size": getattr(getattr(agent, "planner", None), "cost_batch_size", None),
        "snapshot": env.get_replay_state(),
        "elapsed_steps": env.elapsed_steps(),
        "pixel_frames": list(env._frames),
        "proprio_frames": list(env._proprios),
        "rng": _rng_state(agent),
    }
    # The single final file is the commit marker: an interruption during saving
    # leaves only a .tmp file, never a falsely completed trial.
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    temp = path / (CHECKPOINT + ".tmp")
    torch.save(payload, temp)
    os.replace(temp, path / CHECKPOINT)
    return payload


def load_episode_checkpoint(path, digest):
    checkpoint = Path(path) / CHECKPOINT
    if not checkpoint.exists():
        return None
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != 1 or payload.get("config_digest") != digest:
        raise ValueError(f"Invalid or incompatible episode checkpoint: {checkpoint}")
    if len(payload["values"]) != 11:
        raise ValueError(f"Incomplete metrics in {checkpoint}")
    return payload


def restore_episode_checkpoint(payload, agent, env):
    env.set_replay_state(payload["snapshot"])
    env.set_elapsed_steps(payload["elapsed_steps"])
    env._frames.clear()
    env._frames.extend(payload["pixel_frames"])
    env._proprios.clear()
    env._proprios.extend(payload["proprio_frames"])
    _restore_rng(payload["rng"], agent)


def legacy_execution(path, cfg):
    """Recognize only fully exported old trials, never partial execution."""
    path = Path(path)
    execution = path / "execution.csv"
    if not execution.exists():
        return None
    with execution.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    success = rows[-1]["success"] == "True"
    if len(rows) != cfg.task_specification.max_episode_steps and not success:
        return None
    suffix = "succ" if success else "fail"
    if not all((path / "visualisation" / f"video_agent_goal_{suffix}.{ext}").is_file() for ext in ("mp4", "pdf")):
        return None
    if [int(row["simulator_step"]) for row in rows] != list(range(1, len(rows) + 1)):
        raise ValueError(f"Noncontiguous recorded steps in {execution}")
    if any(row["success"] == "True" for row in rows[:-1]):
        raise ValueError(f"Recorded trial did not stop at first success: {execution}")
    return rows


def recover_legacy_episode(path, cfg, agent, env, evaluator, task_idx, digest):
    """Reproduce task metrics and advance CEM's original continuous RNG stream."""
    rows = legacy_execution(path, cfg)
    if rows is None:
        return None
    if cfg.task_specification.task != "mw-reach-wall" or cfg.planner.planner_name != "cem":
        raise ValueError("Legacy recovery currently supports Reach-Wall CEM only")
    if cfg.logging.optional_plots or cfg.planner.decode_each_iteration:
        raise ValueError("Legacy recovery requires the original no-decoder, no-optional-plots A config")
    path = Path(path)
    with (path / "branch_a.csv").open(newline="", encoding="utf-8") as handle:
        captures = list(csv.DictReader(handle))
    calls = sorted({int(row["plan_call"]) for row in rows})
    if calls != list(range(len(calls))):
        raise ValueError(f"Noncontiguous plan calls in {path}")
    for call in calls:
        captured = [row for row in captures if int(row["plan_call"]) == call]
        if len(captured) != cfg.planner.iterations * (cfg.planner.num_samples + 1):
            raise ValueError(f"Incomplete candidate capture in {path}, call {call}")
        if not (path / "branch_a" / f"plan_{call:03d}.pt").exists():
            raise ValueError(f"Missing candidate bundle in {path}, call {call}")
        horizons = {int(row["horizon"]) for row in captured}
        if len(horizons) != 1:
            raise ValueError(f"Mixed horizons in {path}, call {call}")
        # Match exactly CEMPlanner.plan's torch.randn shape and CUDA generator.
        for _ in range(cfg.planner.iterations):
            torch.randn(next(iter(horizons)), cfg.planner.num_samples, agent.model.action_dim,
                        device=agent.local_gpu_generator.device, generator=agent.local_gpu_generator)

    first = torch.load(path / "branch_a" / "plan_000.pt", map_location="cpu", weights_only=False)
    ep = int(path.name.removeprefix("ep_"))
    expected_seed = (cfg.local_seed * cfg.local_seed + ep * cfg.local_seed) % (2**32 - 2)
    if first["episode"] != ep or first["episode_seed"] != expected_seed:
        raise ValueError(f"Saved episode seed does not match the requested config: {path}")
    env.set_replay_state(first["simulator_snapshot"])
    # The original expert is a deterministic scripted controller for this task.
    expert_obs, _, _, expert_infos = evaluator.get_goal_state_mw(env, task_idx=task_idx)
    np.testing.assert_allclose(expert_obs[-1]["proprio"], first["goal_observation"]["proprio"], atol=1e-6, rtol=0)
    expert_state = expert_infos[-1]["state"]
    env.set_replay_state(first["simulator_snapshot"])
    env.set_elapsed_steps(first["episode_steps_elapsed"])
    env._frames.clear()
    env._frames.extend(first["pixel_frames"])
    env._proprios.clear()
    env._proprios.extend(first["proprio_frames"])
    reward_sum = 0.0
    for index, row in enumerate(rows):
        action = torch.tensor([float(row[f"action_{d}"]) for d in range(4)], dtype=torch.float32)
        _, reward, done, truncated, info = env.step(action)
        reward_sum += float(reward)
        if abs(float(info["obj_to_target"]) - float(row["tcp_goal_distance_m"])) > 1e-8:
            raise ValueError(f"Simulator replay disagrees with recorded distance at {path}, step {index + 1}")
        if bool(info["success"]) != (row["success"] == "True"):
            raise ValueError(f"Simulator replay disagrees with recorded success in {path}")
        if (done or truncated) and index != len(rows) - 1:
            raise ValueError(f"Simulator replay terminates too early in {path}")
    end_distance = torch.norm(info["proprio"][-1, :3] - first["goal_observation"]["proprio"][-1, :3]).item()
    values = (expert_infos[-1]["success"], info["success"], reward_sum, float(end_distance < .05),
              end_distance, -1., -1., -1., np.linalg.norm(expert_state - info["state"]), -1., -1.)
    # No original wall-clock duration was persisted. Preserve missing timing
    # explicitly instead of inventing a duration from file timestamps or replay.
    last = torch.load(path / "branch_a" / f"plan_{calls[-1]:03d}.pt", map_location="cpu", weights_only=False)
    random.setstate(last["rng"]["python"])
    np.random.set_state(last["rng"]["numpy"])
    torch.set_rng_state(last["rng"]["torch_cpu"])
    log.info("Recovered completed %s by simulator replay; original duration unavailable", path)
    return save_episode_checkpoint(path, digest, agent, env, values, float("nan"), source="legacy_simulator_replay")


def archive_incomplete_episode(path):
    path = Path(path).resolve()
    if not path.exists() or not any(path.iterdir()):
        return None
    target = path.parent / (path.name + ".interrupted_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    # Both resolved paths must stay in this episode's explicitly named parent.
    if target.resolve().parent != path.parent or path.parent == path:
        raise ValueError("Invalid partial-episode archive path")
    path.rename(target)
    log.info("Archived incomplete episode to %s", target)
    return target

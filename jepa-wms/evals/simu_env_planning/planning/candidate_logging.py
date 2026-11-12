"""Save a fixed candidate pool for later simulator replay and A/B/C scoring."""

import copy
import csv
import random

import numpy as np
import torch
from einops import rearrange
from tensordict import TensorDict


def _cpu_copy(value):
    """Keep saved bundles independent of mutable tensors and GPU state."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, (dict, TensorDict)):
        return {key: _cpu_copy(item) for key, item in value.items()}
    return copy.deepcopy(value)


def save_branch_a_candidates(agent, env, obs, goal_obs, cfg, work_dir, ep, ep_seed, plan_call, executed):
    candidates = agent._planning_info.get("diagnostic_candidates", [])
    if not candidates:
        return []
    task = cfg.task_specification.task
    if not (task.startswith("wall-") or task == "mw-reach-wall"):
        raise ValueError("Candidate replay snapshots currently support Wall and reach-wall")

    native = env.proprio_env.unwrapped
    saved_candidates, rows = [], []
    for candidate in candidates:
        candidate = _cpu_copy(candidate)
        actions = rearrange(candidate["actions_normalized"], "t (f d) -> (t f) d", d=env.action_dim)
        if cfg.planner.repeat_actskip:
            actions = actions.repeat_interleave(agent.model.action_skip, dim=0)
        if agent.preprocessor is not None:
            actions = agent.preprocessor.denormalize_actions(actions)
        candidate["actions_simulator"] = _cpu_copy(actions)
        candidate["is_executed_sequence"] = torch.equal(candidate["actions_normalized"], executed.cpu())
        candidate["is_executed_prefix"] = torch.equal(
            candidate["actions_normalized"][:len(executed)], executed.cpu()
        )
        saved_candidates.append(candidate)
        rows.append({
            "task": task,
            "episode": ep,
            "episode_seed": ep_seed,
            "plan_call": plan_call,
            "candidate": candidate["name"],
            "cem_round": candidate["cem_round"],
            "horizon": candidate["actions_normalized"].shape[0],
            "simulator_steps": len(actions),
            "a_cost": candidate["a_cost"],
            "is_executed_sequence": candidate["is_executed_sequence"],
            "is_executed_prefix": candidate["is_executed_prefix"],
            "origin": candidate.get("origin", "updated_mean"),
            "sample_index": candidate.get("sample_index"),
            "is_elite": candidate.get("is_elite"),
            "is_deployed_output": candidate.get(
                "is_deployed_output", False
            ),
        })

    # Taken after planning, immediately before simulator execution. Restoring
    # these RNG states lets later replay reproduce stochastic collision handling.
    bundle = {
        "schema_version": 2 if task.startswith("mw-") else 1,
        "task": task,
        "episode": ep,
        "episode_seed": ep_seed,
        "plan_call": plan_call,
        "checkpoint": str(cfg.model_kwargs.checkpoint),
        "execution_settings": {"cost_batch_size": cfg.planner.get("cost_batch_size")},
        "initial_observation": _cpu_copy(obs),
        "initial_state": _cpu_copy(obs["proprio"][-1]),
        "initial_encoding": _cpu_copy(agent._planning_info["initial_encoding"]),
        "goal_observation": _cpu_copy(goal_obs),
        "goal_state": _cpu_copy(goal_obs["proprio"][-1]),
        "goal_encoding": _cpu_copy(agent.goal_state_enc),
        "latent_objective": {
            "type": cfg.planner.planning_objective.objective_type,
            "alpha": agent.objective.alpha,
            "sum_all_diffs": agent.objective.sum_all_diffs,
        },
        "action_mean": _cpu_copy(agent.preprocessor.action_mean),
        "action_std": _cpu_copy(agent.preprocessor.action_std),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
        },
        "candidates": saved_candidates,
    }
    if task.startswith("wall-"):
        bundle["wall"] = {
            name: _cpu_copy(getattr(native, name))
            for name in ("wall_x", "hole_y", "left_wall_x", "right_wall_x")
        }
        bundle["wall_config"] = copy.deepcopy(vars(native.wall_config))
    else:
        bundle["simulator_snapshot"] = env.get_replay_state()
        bundle["initial_state"] = _cpu_copy(bundle["simulator_snapshot"]["last_state"])
        bundle["simulator_goal_xyz"] = _cpu_copy(native._target_pos)
        bundle["episode_steps_elapsed"] = env.elapsed_steps()
        bundle["episode_steps_left"] = env.steps_left()
        bundle["pixel_frames"] = _cpu_copy(list(env._frames))
        bundle["proprio_frames"] = _cpu_copy(list(env._proprios))
        bundle["returned_actions_normalized"] = _cpu_copy(executed)
        bundle["action_low"] = _cpu_copy(env.action_space.low)
        bundle["action_high"] = _cpu_copy(env.action_space.high)
        bundle["task_stopping"] = {
            "native_success": True,
            "done_at_succ": cfg.task_specification.done_at_succ,
            "max_episode_steps": cfg.task_specification.max_episode_steps,
        }
        for candidate in saved_candidates:
            raw = candidate["actions_simulator"]
            if not (torch.isfinite(raw).all() and (raw >= torch.as_tensor(bundle["action_low"]) - 1e-6).all()
                    and (raw <= torch.as_tensor(bundle["action_high"]) + 1e-6).all()):
                raise ValueError("Captured Meta-World candidate violates the benchmark action bounds")
    output_dir = work_dir / "branch_a"
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, output_dir / f"plan_{plan_call:03d}.pt")
    csv_path = work_dir / "branch_a.csv"
    with csv_path.open("w" if plan_call == 0 else "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        if plan_call == 0:
            writer.writeheader()
        writer.writerows(rows)
    return rows

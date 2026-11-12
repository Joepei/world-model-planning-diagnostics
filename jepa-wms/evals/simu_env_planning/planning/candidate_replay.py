from pathlib import Path
import logging
import hashlib
# Hide verbose setup logs (dataset indices, model structures, etc.).
# Warnings, errors, and the explicit result prints below remain visible.
logging.disable(logging.INFO)
import csv
import torch
import random

import numpy as np
from omegaconf import OmegaConf

from evals.simu_env_planning.envs.init import make_env
from evals.simu_env_planning.planning.common.parser import parse_cfg

from tensordict import TensorDict

from evals.utils import make_datasets
from app.vjepa_wm.modelcustom.simu_env_planning.vit_enc_preds import init_module
from evals.simu_env_planning.planning.planning.objectives import (
    ReprTargetDistMPCObjective,
)

def remaining_route_distance(position, goal, wall):
    left, right = float(wall.left_wall_x), float(wall.right_wall_x)
    low = float(wall.hole_y) - wall.wall_config.door_space
    high = float(wall.hole_y) + wall.wall_config.door_space

    points = np.array([
        position.detach().cpu().tolist(),
        goal.detach().cpu().tolist(),
        [left, low], [left, high],
        [right, low], [right, high],
    ], dtype=float)

    def clear_segment(a, b):
        dx = b[0] - a[0]
        if abs(dx) < 1e-12:
            return not (left < a[0] < right) or (
                min(a[1], b[1]) >= low
                and max(a[1], b[1]) <= high
            )

        t1, t2 = (left - a[0]) / dx, (right - a[0]) / dx
        enter = max(0.0, min(t1, t2))
        leave = min(1.0, max(t1, t2))
        if enter >= leave:
            return True

        y1 = a[1] + enter * (b[1] - a[1])
        y2 = a[1] + leave * (b[1] - a[1])
        return (
            min(y1, y2) >= low - 1e-9
            and max(y1, y2) <= high + 1e-9
        )

    distances = np.full((6, 6), np.inf)
    for i, a in enumerate(points):
        for j, b in enumerate(points):
            if clear_segment(a, b):
                distances[i, j] = np.linalg.norm(a - b)

    # Find the shortest combination of unobstructed segments.
    for k in range(6):
        distances = np.minimum(
            distances,
            distances[:, k, None] + distances[None, k, :],
        )

    return float(distances[0, 1])

# Find the project directory independently of the terminal's location.
project_root = Path(__file__).resolve().parents[4]

run_dir = (
    project_root
    / "runs/wall/simu_env_planning/pool650_i10"
)



"""Replay the saved actions 
"""

# Load the configuration used for the original run.
cfg = OmegaConf.load(run_dir / "args_eval.yaml")
cfg.work_dir = str(run_dir)
cfg.frameskip = cfg.model_kwargs.data.custom.frameskip
cfg = parse_cfg(cfg)

# Create the environment. "wall" accesses its underlying simulator.
env = make_env(cfg)
wall = env.proprio_env.unwrapped

"""
Add scoring by executing actions in the dynamic simulator
"""
# Match the numerical settings used in the original evaluation.
torch.backends.cudnn.benchmark = True

settings = OmegaConf.to_container(cfg.model_kwargs, resolve=True)

# Recover the original image transforms and normalization statistics.
dset, preprocessor = make_datasets(
    settings["data"],
    settings["data_aug"],
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = init_module(
    folder=cfg.folder,
    checkpoint=settings["checkpoint"],
    model_kwargs=settings["pretrain_kwargs"],
    device=device,
    action_dim=dset.action_dim,
    proprio_dim=dset.proprio_dim,
    preprocessor=preprocessor,
    cfgs_data=settings["data"],
    wrapper_kwargs=settings.get("wrapper_kwargs", {}),
)
model.eval()

# Find all saved episode bundles and sort by episode number.
bundle_paths = sorted(
    run_dir.glob("wall-base/ep_*/branch_a/plan_000.pt"),
    key=lambda path: int(path.parents[1].name.split("_")[-1]),
)

all_results = []

for bundle_path in bundle_paths:
    bundle = torch.load(
        bundle_path,
        map_location="cpu",
        weights_only=False,
    )

    print(f"\n=== Episode {bundle['episode']} ===")

    for key, value in bundle["wall_config"].items():
        setattr(wall.wall_config, key, value)

    for key, value in bundle["wall"].items():
        setattr(wall, key, value.clone())

    results = []


    for candidate in bundle["candidates"]:
        actions = candidate["actions_simulator"]
        print(f"\nEvaluating {candidate['name']}")

        # Restore the dot's starting position and reset the step counter.
        obs, info = env.prepare(
            bundle["episode_seed"],
            bundle["initial_state"].clone(),
        )

        # Confirm that the restored position and image match the saved start.
        torch.testing.assert_close(info["state"], bundle["initial_state"])
        torch.testing.assert_close(
            obs.float(),
            bundle["initial_observation"]["visual"].float(),
        )

        # Restore randomness after setup, immediately before execution.
        random.setstate(bundle["rng"]["python"])
        np.random.set_state(bundle["rng"]["numpy"])
        torch.set_rng_state(bundle["rng"]["torch_cpu"])

        # Task execution: stop at the first goal hit.
        observations, rewards, dones, infos = [], [], [], []
        task_state = bundle["initial_state"].clone()

        for action in actions:
            if torch.linalg.vector_norm(
                task_state - bundle["goal_state"]
            ).item() < 4.5:
                break

            obs_part, reward_part, done_part, info_part = env.step_multiple(
                action.unsqueeze(0)
            )
            observations.extend(obs_part)
            rewards.extend(reward_part)
            dones.extend(done_part)
            infos.extend(info_part)

            task_state = infos[-1]["state"].clone()
            if dones[-1]:
                break

        task_steps = len(infos)
        task_end_distance = torch.linalg.vector_norm(
            task_state - bundle["goal_state"]
        ).item()
        task_success = task_end_distance < 4.5

        # Task has ended. Remaining actions are diagnostic only.
        if task_steps < len(actions) and not (dones and dones[-1]):
            obs_part, reward_part, done_part, info_part = env.step_multiple(
                actions[task_steps:]
            )
            observations.extend(obs_part)
            rewards.extend(reward_part)
            dones.extend(done_part)
            infos.extend(info_part)

        final_state = infos[-1]["state"]
        # Include the starting position as step 0.
        states = torch.stack(
            [bundle["initial_state"]]
            + [item["state"] for item in infos]
        )

        distances_to_goal = torch.linalg.vector_norm(
            states - bundle["goal_state"],
            dim=-1,
        )

        # Match the environment's success threshold.
        hit_steps = torch.nonzero(
            distances_to_goal < 4.5
        ).flatten()

        first_hit = int(hit_steps[0]) if len(hit_steps) else None

        remaining = remaining_route_distance(
            task_state,
            bundle["goal_state"],
            wall,
        )
        goal_distance = torch.linalg.vector_norm(
            final_state - bundle["goal_state"]
        ).item()

        # print("Executed actions:", len(observations))
        # print("Actual endpoint:", final_state)
        # print("Final goal distance:", goal_distance)
        print("Task actions:", task_steps)
        print("Task endpoint:", task_state)
        print("Task final goal distance:", task_end_distance)


        # The encoder expects [batch, time, ...].
        # These observations already have a time dimension;
        # unsqueeze(0) adds a batch containing one example.
        final_obs = {
            "visual": observations[-1].unsqueeze(0),
            "proprio": infos[-1]["proprio"].unsqueeze(0),
        }

        with torch.inference_mode():
            actual_enc = model.encode(final_obs)

            # Check that loading and preprocessing reproduce the saved goal.
            checked_goal = model.encode({
                key: value.unsqueeze(0)
                for key, value in bundle["goal_observation"].items()
            })

        for key in ("visual", "proprio"):
            torch.testing.assert_close(
                checked_goal[key].cpu(),
                bundle["goal_encoding"][key],
                rtol=1e-4,
                atol=1e-5,
            )

        print("Goal encoding check passed.")


        goal_enc = TensorDict(
            {
                key: value.to(device)
                for key, value in bundle["goal_encoding"].items()
            },
            batch_size=[],
        )

        objective = ReprTargetDistMPCObjective(
            cfg,
            target_enc=goal_enc,
            alpha=bundle["latent_objective"]["alpha"],
            sum_all_diffs=bundle["latent_objective"]["sum_all_diffs"],
        )

        # Encoder output uses [batch, time, ...].
        # The objective expects [time, batch, ...].
        actual_enc = TensorDict(
            {
                key: value.transpose(0, 1)
                for key, value in actual_enc.items()
            },
            batch_size=[],
        )

        normalized_actions = (
            candidate["actions_normalized"].to(device).unsqueeze(1)
        )

        b_cost = objective(actual_enc, normalized_actions).item()

        print("A cost — predicted endpoint:", candidate["a_cost"])
        print("B cost — actual endpoint:   ", b_cost)
        results.append({
            "episode": bundle["episode"],
            "candidate": candidate["name"],
            "cem_round": candidate["cem_round"],
            "is_executed_sequence": candidate["is_executed_sequence"],
            "a_cost": candidate["a_cost"],
            "b_cost": b_cost,
            "end_distance": goal_distance,
            "c_success":  task_success,
            "task_steps": task_steps,
            "task_end_distance": task_end_distance,
            "c_first_hit_step": first_hit,
            "c_remaining_distance": remaining,
            "origin": candidate["origin"],
            "sample_index": candidate["sample_index"],
            "is_elite": candidate["is_elite"],
            "is_deployed_output": candidate["is_deployed_output"],
            "route_id": hashlib.sha256(
                candidate["actions_normalized"]
                .contiguous().numpy().tobytes()
            ).hexdigest(),
        })

    best_a = min(results, key=lambda row: row["a_cost"])
    best_b = min(results, key=lambda row: row["b_cost"])
    best_c = min(
    results,
    key=lambda row: (
        (0, row["c_first_hit_step"])
        if row["c_success"]
        else (1, row["c_remaining_distance"])
        ),
    )

    

    print("\nLowest A cost:", best_a)
    print("\nLowest B cost:", best_b)
    print("\nLowest C cost:", best_c)
    all_results.extend(results)


# output_path = run_dir / "branch_b.csv"
# output_path = run_dir / "branch_abc.csv"
# output_path = run_dir / "branch_abc_stop.csv"
output_path = run_dir / "scored_candidates.csv"


with output_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=list(all_results[0]),
    )
    writer.writeheader()
    writer.writerows(all_results)

print(f"\nSaved {len(all_results)} candidate scores to:")
print(output_path)
env.close()

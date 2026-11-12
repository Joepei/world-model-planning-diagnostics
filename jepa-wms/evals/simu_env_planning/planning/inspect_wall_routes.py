"""Inspect saved Wall routes and the environmental context of latent ranking errors.

Replays captured actions in the original simulator, without loading a world
model. Full-horizon trajectories are diagnostic, including after first success.
Run: python -m evals.simu_env_planning.planning.inspect_wall_routes
"""

import argparse
import csv
import hashlib
import html
import json
import logging
from pathlib import Path
import random
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import matplotlib.patheffects as pe
import numpy as np
from omegaconf import OmegaConf
from scipy.stats import spearmanr
import torch

from .analyze_candidates import deduplicate_routes, group_by, load_scores, write_csv
from .common.parser import parse_cfg


DEFAULT_RUN = Path(__file__).resolve().parents[4] / "runs/wall/simu_env_planning/pool650_i10"
METHODS = {
    "deployed": ("A deployed mean", "#9B51A0", "P"),
    "min_b": ("B minimum latent cost", "#0072B2", "X"),
    "min_terminal": ("Minimum terminal distance", "#009E73", "D"),
}


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def restore(env, bundle, check_image=False):
    native = env.proprio_env.unwrapped
    for key, value in bundle["wall_config"].items():
        setattr(native.wall_config, key, value)
    for key, value in bundle["wall"].items():
        setattr(native, key, value.clone())
    if check_image:
        obs, info = env.prepare(bundle["episode_seed"], bundle["initial_state"].clone())
        torch.testing.assert_close(obs.float(), bundle["initial_observation"]["visual"].float(), rtol=0, atol=0)
        torch.testing.assert_close(info["state"], bundle["initial_state"], rtol=0, atol=0)
    else:
        native.set_init_state(bundle["initial_state"].clone())
        native.reset()
    random.setstate(bundle["rng"]["python"])
    np.random.set_state(bundle["rng"]["numpy"])
    torch.set_rng_state(bundle["rng"]["torch_cpu"])
    return native


def replay(env, bundle, candidate, wrapped=False):
    native = restore(env, bundle, check_image=wrapped)
    states = [bundle["initial_state"].clone()]
    actions = candidate["actions_simulator"]
    if wrapped:
        observations, _, _, infos = env.step_multiple(actions)
        states.extend(info["state"].clone() for info in infos)
        image = observations[-1][-1].permute(1, 2, 0).numpy()
    else:
        # The same DotWall.step used by the wrapper: real collision physics and
        # native rendering, omitting only repeated 224px resizing/history copies.
        for action in actions:
            _, _, _, info = native.step(action.numpy())
            states.append(info["state"].clone())
        image = None
    states = torch.stack(states)
    if len(states) != len(actions) + 1:
        raise ValueError("Replay did not cover the complete saved action horizon")
    return states, image


def check_outcome(states, row, bundle):
    distances = torch.linalg.vector_norm(states - bundle["goal_state"], dim=-1)
    error = abs(float(distances[-1]) - row["end_distance"])
    if error > 1e-5:
        raise ValueError(f"Saved endpoint mismatch: episode {row['episode']}, {row['candidate']}, error={error}")
    hits = torch.nonzero(distances < 4.5).flatten()
    first_hit = int(hits[0]) if len(hits) else None
    if bool(len(hits)) != row["c_success"] or first_hit != row["c_first_hit_step"]:
        raise ValueError("Replay first-hit outcome disagrees with the saved experiment")
    return error


def region(x, bundle):
    left, right = (float(bundle["wall"][name]) for name in ("left_wall_x", "right_wall_x"))
    goal_x = float(bundle["goal_state"][0])
    if left <= x <= right:
        return "doorway"
    same_goal_side = (x < left and goal_x < left) or (x > right and goal_x > right)
    return "goal side" if same_goal_side else "start side"


def background(ax, bundle, native, label=True):
    wall = bundle["wall"]
    native.wall_config.img_size = bundle["wall_config"]["img_size"]
    mask = native._render_walls(wall["wall_x"], wall["hole_y"]).numpy()
    n = mask.shape[0]
    ax.imshow(mask, cmap="Greys", vmin=0, vmax=1.6, interpolation="nearest",
              extent=(-.5, n - .5, n - .5, -.5), zorder=0)
    start, goal = bundle["initial_state"].numpy(), bundle["goal_state"].numpy()
    ax.add_patch(Circle(goal, 4.5, facecolor="#F6C344", alpha=.16, edgecolor="#BC8500", linewidth=.9))
    ax.scatter(*start, c="black", s=35, marker="o", edgecolors="white", linewidths=.7, zorder=6)
    ax.scatter(*goal, c="#F6C344", s=135, marker="*", edgecolors="black", linewidths=.6, zorder=7)
    ax.set(xlim=(-.5, n - .5), ylim=(n - .5, -.5), aspect="equal", xticks=[0, 16, 32, 48, 64], yticks=[0, 16, 32, 48, 64])
    ax.tick_params(labelsize=8)
    if label:
        ax.set_xlabel("x (Wall coordinates)", fontsize=9)
        ax.set_ylabel("y (increases downward)", fontsize=9)


def route_line(ax, states, color, linewidth=1.7):
    xy = states.numpy()
    ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=linewidth, alpha=.9, zorder=3)
    ax.scatter(xy[1:, 0], xy[1:, 1], color=color, s=9, alpha=.65, zorder=4)
    for step in range(5, len(xy), 5):
        ax.annotate("", xy=xy[step], xytext=xy[step - 1],
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.2), zorder=5)
    ax.scatter(*xy[-1], marker="X", s=95, c=color, edgecolors="white", linewidths=.7, zorder=8)


def save(fig, output, name):
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"{name}.{suffix}", dpi=210, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def plot_focused_routes(episodes, bundles, routes, selected, native, output):
    fig, axes = plt.subplots(len(episodes), 3, figsize=(12, 4.2 * len(episodes)), squeeze=False)
    for i, ep in enumerate(episodes):
        for j, (name, (label, color, _)) in enumerate(METHODS.items()):
            ax, row = axes[i, j], selected[ep, name]
            background(ax, bundles[ep], native)
            route_line(ax, routes[ep, name], color)
            ax.set_title(f"Episode {ep} | {label}\nB={float(row['b_cost']):.4f}; terminal d={float(row['end_distance']):.2f}", fontsize=10)
    fig.suptitle("Exact simulator routes from the shared saved pool", fontsize=15)
    fig.text(.5, .012, "Black circle: start   |   Gold star/disk: goal and success radius   |   Colored X: endpoint   |   Arrows every 5 raw actions\n"
             "All routes show the complete 30-action diagnostic horizon. None of these episode-3/4 routes succeeds.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .055, 1, .96))
    save(fig, output, "routes_episodes_3_4")


def plot_candidate_routes(episodes, bundles, pool_routes, routes, selected, native, output):
    """Show every unique route, with the two terminal selectors above the pool."""
    fig, axes = plt.subplots(1, len(episodes), figsize=(6.3 * len(episodes), 6.8), squeeze=False)
    for ax, ep in zip(axes.flat, episodes):
        background(ax, bundles[ep], native)
        paths = [states.numpy() for (episode, _), states in pool_routes.items() if episode == ep]
        ax.add_collection(LineCollection(paths, colors="#666666", linewidths=.55,
                                         alpha=.045, zorder=2, rasterized=True))
        for name, color, marker in (("min_terminal", "#009E73", "D"), ("min_b", "#0072B2", "X")):
            xy = routes[ep, name].numpy()
            line, = ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=2.4, zorder=5)
            line.set_path_effects([pe.Stroke(linewidth=4.1, foreground="white"), pe.Normal()])
            for step in range(5, len(xy), 5):
                ax.annotate("", xy=xy[step], xytext=xy[step - 1],
                            arrowprops=dict(arrowstyle="->", color=color, lw=1.5), zorder=6)
            ax.scatter(*xy[-1], marker=marker, s=135, c=color, edgecolors="white", linewidths=1, zorder=9)
        # Keep the shared start visible over both highlighted paths.
        ax.scatter(*bundles[ep]["initial_state"].numpy(), c="black", s=45,
                   edgecolors="white", linewidths=.8, zorder=10)
        b_distance = float(selected[ep, "min_b"]["end_distance"])
        c_distance = float(selected[ep, "min_terminal"]["end_distance"])
        ax.set_title(f"Episode {ep}: all {len(paths)} unique routes\n"
                     f"B terminal distance = {b_distance:.2f}    |    C terminal distance = {c_distance:.2f}", fontsize=11, pad=12)
    handles = [
        Line2D([], [], color="#888888", linewidth=1, label="All candidate routes"),
        Line2D([], [], color="#0072B2", linewidth=2.4, marker="X", label="B pick: minimum terminal latent cost"),
        Line2D([], [], color="#009E73", linewidth=2.4, marker="D", label="C pick: minimum terminal physical distance"),
        Line2D([], [], color="black", marker="o", linestyle="None", label="Start"),
        Line2D([], [], color="#F6C344", marker="*", markeredgecolor="black", markersize=11,
               linestyle="None", label="Goal"),
    ]
    fig.suptitle("All captured candidate routes with terminal B/C selections", fontsize=15)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .035),
               ncol=3, frameon=False, fontsize=9)
    fig.text(.5, .012, "Gray density reflects overlap, not probability. Each route contains 30 simulator actions; duplicates are drawn once.",
             ha="center", fontsize=9, color=".3")
    fig.tight_layout(rect=(0, .135, 1, .94), w_pad=2)
    save(fig, output, "all_candidate_routes_episodes_3_4")


def plot_endpoint_alignment(episodes, bundles, endpoints, selected, native, output):
    fig, axes = plt.subplots(len(episodes), 2, figsize=(12, 4.8 * len(episodes)), squeeze=False)
    for i, ep in enumerate(episodes):
        rows = [row for row in endpoints if row["episode"] == ep]
        b = np.array([row["b_cost"] for row in rows])
        d = np.array([row["end_distance"] for row in rows])
        xy = np.array([[row["x"], row["y"]] for row in rows])
        ax = axes[i, 0]
        background(ax, bundles[ep], native)
        dots = ax.scatter(xy[:, 0], xy[:, 1], c=b, cmap="viridis", s=19, alpha=.65, zorder=2)
        fig.colorbar(dots, ax=ax, shrink=.8, pad=.025, label="B latent cost (lower is better)")
        ax.set_title(f"Episode {ep}: {len(rows)} actual candidate endpoints", fontsize=11)
        scatter = axes[i, 1]
        colors = {"start side": "#A7A7A7", "doorway": "#E69F00", "goal side": "#56B4E9"}
        for label, color in colors.items():
            group = [row for row in rows if row["region"] == label]
            if group:
                scatter.scatter([row["end_distance"] for row in group], [row["b_cost"] for row in group],
                                s=18, alpha=.55, c=color, label=label, edgecolors="none")
        lookup = {row["route_id"]: row for row in rows}
        for name, (_, color, marker) in METHODS.items():
            chosen = selected[ep, name]
            row = lookup[chosen["route_id"]]
            ax.scatter(row["x"], row["y"], s=140, marker=marker, c=color, edgecolors="white", linewidths=1, zorder=9)
            scatter.scatter(row["end_distance"], row["b_cost"], s=130, marker=marker,
                            c=color, edgecolors="black", linewidths=.6, zorder=8)
        rho = float(spearmanr(b, d).statistic)
        scatter.set(title=f"Episode {ep}: B versus terminal distance | rho={rho:.3f}",
                    xlabel="Actual terminal goal distance (lower is better)", ylabel="B latent cost (lower is better)")
        scatter.grid(alpha=.18)
        scatter.legend(loc="best", fontsize=8, title="Endpoint region", title_fontsize=9)
    handles = [Line2D([], [], color=color, marker=marker, linestyle="None", markersize=8, label=label)
               for label, color, marker in METHODS.values()]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=10)
    fig.suptitle("Where does the actual-endpoint latent cost disagree?", fontsize=15)
    fig.tight_layout(rect=(0, .045, 1, .96), w_pad=2)
    save(fig, output, "endpoint_alignment_episodes_3_4")


def plot_all_routes(bundles, routes, geometry, native, output):
    episodes = sorted(bundles)
    fig, axes = plt.subplots(2, 5, figsize=(16, 8.5), squeeze=False)
    for ax, ep in zip(axes.flat, episodes):
        background(ax, bundles[ep], native, label=False)
        for name in ("min_terminal", "min_b"):
            route_line(ax, routes[ep, name], METHODS[name][1], linewidth=1.25)
        record = next(row for row in geometry if row["episode"] == ep)
        ax.set_title(f"Episode {ep} | rho={record['spearman_b_terminal']:.2f}\nB regret={record['b_terminal_regret']:.2f}", fontsize=10)
        if ep in (3, 4):
            for spine in ax.spines.values():
                spine.set_color("#C0392B")
                spine.set_linewidth(2)
    fig.suptitle("The same wall and doorway, ten different start/goal placements", fontsize=15)
    fig.text(.5, .015, "Blue: minimum B   |   Green: minimum terminal distance   |   Black circle: start   |   Gold star: goal\n"
             "Full-horizon simulator replay; overlapping routes can hide one another. Episode cases are correlated candidate pools, not independent routes.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .08, 1, .94), h_pad=3.0, w_pad=1.1)
    save(fig, output, "selected_routes_all_episodes")


def plot_images(episodes, bundles, images, selected, output):
    fig, axes = plt.subplots(len(episodes), 3, figsize=(11, 3.7 * len(episodes)), squeeze=False)
    for i, ep in enumerate(episodes):
        goal = bundles[ep]["goal_observation"]["visual"][-1].permute(1, 2, 0).numpy()
        for j, (label, image) in enumerate((("Goal image", goal),
                ("B-selected endpoint", images[ep, "min_b"]),
                ("Terminal-best endpoint", images[ep, "min_terminal"]))):
            ax = axes[i, j]
            ax.imshow(np.asarray(image).clip(0, 255).astype(np.uint8), interpolation="nearest")
            ax.set_title(f"Episode {ep} | {label}", fontsize=11)
            ax.axis("off")
    fig.suptitle("Actual images: the dot location changes; wall geometry is fixed", fontsize=14)
    fig.text(.5, .015, "Goal images are copied from the saved bundles; endpoint images come from exact wrapped replay.\n"
             "These are simulator observations, not decoded world-model predictions. B also uses a proprioceptive latent component.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .065, 1, .95))
    save(fig, output, "input_images_episodes_3_4")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    run = args.run_dir
    output = args.output_dir or run / "analysis_terminal/route_inspection"
    output.mkdir(parents=True, exist_ok=True)
    logging.disable(logging.INFO)
    warnings.filterwarnings("ignore", message="To copy construct from a tensor")
    torch.set_num_threads(1)
    from evals.simu_env_planning.envs.init import make_env
    cfg = OmegaConf.load(run / "args_eval.yaml")
    cfg.work_dir = str(run)
    cfg.frameskip = cfg.model_kwargs.data.custom.frameskip
    env = make_env(parse_cfg(cfg))
    native = env.proprio_env.unwrapped
    rows = load_scores(run / "scored_candidates.csv")
    episode_rows = group_by(rows, "episode")
    selected = {(int(row["episode"]), row["selection"]): row
                for row in read_csv(run / "analysis_terminal/selections.csv")}
    episode_metrics = {int(row["episode"]): row for row in read_csv(run / "analysis_terminal/episode_summary.csv")}
    focused = (3, 4)
    bundles, routes, images, geometry, endpoints, trajectory_rows = {}, {}, {}, [], [], []
    pool_routes, pool_metadata = {}, []
    maximum_error = 0.
    hashes = {}
    try:
        for ep, occurrences in sorted(episode_rows.items()):
            path = run / f"wall-base/ep_{ep}/branch_a/plan_000.pt"
            hashes[str(path.relative_to(run))] = hashlib.sha256(path.read_bytes()).hexdigest()
            bundle = torch.load(path, map_location="cpu", weights_only=False)
            bundles[ep] = bundle
            candidates = {candidate["name"]: candidate for candidate in bundle["candidates"]}
            unique = deduplicate_routes(occurrences)
            by_name = {row["candidate"]: row for row in occurrences}
            for name in METHODS:
                row = selected[ep, name]
                candidate = candidates[row["candidate"]]
                states, image = replay(env, bundle, candidate, wrapped=True)
                maximum_error = max(maximum_error, check_outcome(states, by_name[row["candidate"]], bundle))
                native_states, _ = replay(env, bundle, candidate)
                torch.testing.assert_close(states, native_states, rtol=0, atol=0)
                routes[ep, name], images[ep, name] = states, image
                for step, state in enumerate(states):
                    trajectory_rows.append(dict(episode=ep, selection=name, candidate=row["candidate"],
                                                step=step, x=float(state[0]), y=float(state[1]),
                                                goal_distance=float(torch.linalg.vector_norm(state - bundle["goal_state"]))))
            start, goal = bundle["initial_state"].numpy(), bundle["goal_state"].numpy()
            b_end = routes[ep, "min_b"][-1].numpy()
            c_end = routes[ep, "min_terminal"][-1].numpy()
            geometry.append(dict(
                episode=ep, start_x=float(start[0]), start_y=float(start[1]),
                goal_x=float(goal[0]), goal_y=float(goal[1]),
                wall_x=float(bundle["wall"]["wall_x"]), door_y=float(bundle["wall"]["hole_y"]),
                goal_vertical_door_offset=abs(float(goal[1] - float(bundle["wall"]["hole_y"]))),
                b_terminal_x=float(b_end[0]), b_terminal_y=float(b_end[1]),
                terminal_best_x=float(c_end[0]), terminal_best_y=float(c_end[1]),
                b_terminal_distance=float(selected[ep, "min_b"]["end_distance"]),
                best_terminal_distance=float(selected[ep, "min_terminal"]["end_distance"]),
                b_terminal_regret=float(selected[ep, "min_b"]["terminal_distance_regret"]),
                spearman_b_terminal=float(episode_metrics[ep]["spearman_b_terminal"]),
            ))
            if ep in focused:
                for row in unique:
                    candidate = candidates[row["candidate"]]
                    route_id = hashlib.sha256(candidate["actions_normalized"].contiguous().numpy().tobytes()).hexdigest()
                    if route_id != row["route_id"]:
                        raise ValueError("Bundle actions disagree with recorded route identity")
                    states, _ = replay(env, bundle, candidate)
                    maximum_error = max(maximum_error, check_outcome(states, row, bundle))
                    pool_routes[ep, route_id] = states
                    pool_metadata.append(row)
                    end = states[-1]
                    endpoints.append(dict(episode=ep, candidate=row["candidate"], route_id=route_id,
                                          x=float(end[0]), y=float(end[1]), region=region(float(end[0]), bundle),
                                          a_cost=row["a_cost"], b_cost=row["b_cost"], end_distance=row["end_distance"],
                                          cem_round=row["cem_round"]))
            print(f"Episode {ep}: selected replay verified" + (f"; {len(unique)} endpoint replays verified" if ep in focused else ""), flush=True)
        with plt.rc_context({"font.size": 10, "pdf.fonttype": 42}):
            plot_candidate_routes(focused, bundles, pool_routes, routes, selected, native, output)
            plot_focused_routes(focused, bundles, routes, selected, native, output)
            plot_endpoint_alignment(focused, bundles, endpoints, selected, native, output)
            plot_all_routes(bundles, routes, geometry, native, output)
            plot_images(focused, bundles, images, selected, output)
    finally:
        env.close()
    write_csv(output / "selected_trajectories.csv", trajectory_rows)
    np.savez_compressed(
        output / "all_candidate_trajectories.npz",
        episode=np.array([row["episode"] for row in pool_metadata]),
        route_id=np.array([row["route_id"] for row in pool_metadata]),
        candidate=np.array([row["candidate"] for row in pool_metadata]),
        states=np.stack([pool_routes[row["episode"], row["route_id"]].numpy() for row in pool_metadata]),
        b_cost=np.array([row["b_cost"] for row in pool_metadata]),
        end_distance=np.array([row["end_distance"] for row in pool_metadata]),
    )
    write_csv(output / "candidate_endpoints.csv", endpoints)
    write_csv(output / "episode_geometry.csv", geometry)
    regret = sum(row["b_terminal_regret"] for row in geometry)
    fraction = sum(row["b_terminal_regret"] for row in geometry if row["episode"] in focused) / regret
    case_metrics = {}
    for ep in focused:
        batch = [row for row in endpoints if row["episode"] == ep]
        case_metrics[ep] = dict(endpoint_regions={label: sum(row["region"] == label for row in batch)
                                                 for label in ("start side", "doorway", "goal side")},
                                b_min_cost=min(row["b_cost"] for row in batch),
                                b_max_cost=max(row["b_cost"] for row in batch))
    summary = dict(focused_episodes=focused, replayed_unique_pool_routes=len(endpoints),
                   cached_full_trajectories=len(pool_routes), states_per_trajectory=31,
                   wrapped_selected_replays=len(routes), maximum_saved_distance_error=maximum_error,
                   selected_native_and_wrapped_states="bitwise equal at every raw step",
                   focused_fraction_of_b_terminal_regret=fraction, cases=case_metrics)
    (output / "inspection_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    protocol = dict(source_scores_sha256=hashlib.sha256((run / "scored_candidates.csv").read_bytes()).hexdigest(),
                    selections_sha256=hashlib.sha256((run / "analysis_terminal/selections.csv").read_bytes()).hexdigest(),
                    bundle_sha256=hashes, simulator="original DotWall.step with native rendering; selected routes also checked through wrappers",
                    replay="saved raw action sequence, wall geometry, start, and Python/NumPy/Torch RNG; no action clipping",
                    costs="original saved A/B scores and full-horizon terminal distance; no new latent scoring",
                    episode_ids="zero-based", diagnostic_horizon_raw_actions=30,
                    hypotheses="descriptive case inspection, not a causal environmental intervention")
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    report = ["# Wall route inspection: episodes 3 and 4", "",
              f"These two zero-based episode IDs account for {fraction:.2%} of B's total terminal-distance regret, "
              "not literally all regret. Small positive regrets also occur in episodes 1, 2, and 5.", "",
              "All 641 unique routes in each focused pool were replayed. Their endpoint distances and first-hit outcomes "
              "match the saved experiment. Thirty selected route replays across all ten episodes were also checked "
              "through the complete observation wrappers, with bitwise-identical state sequences.", "",
              "## Focused observations", "",
              "| Episode | B endpoint | Terminal-best endpoint | B distance | Best distance | B regret | B/terminal rho |",
              "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for row in geometry:
        if row["episode"] in focused:
            report.append(f"| {row['episode']} | ({row['b_terminal_x']:.2f}, {row['b_terminal_y']:.2f}) | "
                          f"({row['terminal_best_x']:.2f}, {row['terminal_best_y']:.2f}) | {row['b_terminal_distance']:.2f} | "
                          f"{row['best_terminal_distance']:.2f} | {row['b_terminal_regret']:.2f} | {row['spearman_b_terminal']:.3f} |")
    report += ["", "- Episode 3: B stays on the starting (left) side below the doorway. The terminal-distance winner reaches the doorway; "
               "it still does not reach the goal. B assigns the doorway winner a higher latent cost.",
               "- Episode 4: B moves toward the upper-right border, away from the lower-left goal. The terminal-distance winner "
               "instead approaches the doorway from the right. Both remain unsuccessful.", "",
               "No candidate in either pool ever reaches the goal, so first-hit timing does not explain the "
               "B/terminal ranking disagreement within these two cases. Episode 3 has 635 endpoints on the start side "
               "and six in the doorway; episode 4 has all 641 endpoints on the start side. No endpoint is on the goal side.", "",
               "## Environmental interpretation", "",
               "All ten saved episodes use the same wall at x=32, thickness spanning x=29..35, with a doorway at y=26..34. "
               "The changed environmental variables in these cases are start and goal positions, hence the endpoint region reached by search.", "",
               "The plots can suggest a failure of latent distance to track spatial displacement when all proposed endpoints are far "
               "from the goal, especially near walls/borders. This is a hypothesis, not an established environmental condition. "
               "Episode 0 also has no successful route but does not have B selection regret; lack of success alone is insufficient.", "",
               "A large goal-to-door vertical offset is also not sufficient: other episodes place the goal away from the doorway "
               "and still show much stronger B/terminal rank agreement. See the all-episode comparison and episode_geometry.csv.", "",
               "The original A-generated pool is confounded with start/goal geometry and reachable endpoint coverage. "
               "The current figures cannot distinguish encoder information loss, its Euclidean readout/weighting, "
               "patch/border effects, or candidate-coverage effects. B includes visual and proprioceptive latent costs; "
               "images alone do not isolate the visual contribution. No out-of-distribution claim is established.", "",
               "A future controlled probe could hold wall/goal fixed and vary dot position over a common valid grid, then "
               "separate visual and proprioceptive costs. That probe was not run here.", "",
               "## Figures", ""]
    names = ("all_candidate_routes_episodes_3_4", "routes_episodes_3_4", "endpoint_alignment_episodes_3_4", "selected_routes_all_episodes", "input_images_episodes_3_4")
    for name in names:
        report.append(f"![{name.replace('_', ' ')}]({name}.png)\n")
    (output / "results.md").write_text("\n".join(report), encoding="utf-8")
    gallery = "<!doctype html><meta charset='utf-8'><title>Wall route inspection</title><style>body{font:17px system-ui;max-width:1350px;margin:30px auto;background:#f5f5f5}img{width:100%;background:white}section{margin:30px 0}a{color:#0072b2}</style><h1>Wall route inspection</h1><p>Exact saved candidate replay. <a href='results.md'>Report</a></p>"
    for name in names:
        gallery += f"<section><h2>{html.escape(name.replace('_', ' ').capitalize())}</h2><a href='{name}.png'><img src='{name}.png'></a></section>"
    (output / "index.html").write_text(gallery, encoding="utf-8")
    print(f"Saved inspection: {output}", flush=True)


if __name__ == "__main__":
    main()

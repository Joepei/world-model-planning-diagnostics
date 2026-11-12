"""Regenerate the main plots from the published measurements, with no simulator.

By default writes figures/reproduced/, preserving the original reviewed exports.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle, Circle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "jepa-wms"))
from evals.simu_env_planning.planning.plot_reach_wall_distance import plot_traces


def rows(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def save(fig, output, name):
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"{name}.{suffix}", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def wall_regret(output):
    source = ROOT / "data/wall/terminal/selections.csv"
    data = rows(source)
    fig, ax = plt.subplots(figsize=(10.4, 4.7))
    episodes = np.arange(10)
    for label, name, color, offset in (("deployed", "A: deployed final mean", "#0072B2", -.18),
                                      ("min_b", "B: minimum simulator-latent cost", "#D55E00", .18)):
        selected = sorted((row for row in data if row["selection"] == label), key=lambda row: int(row["episode"]))
        ax.bar(episodes + offset, [float(row["terminal_distance_regret"]) for row in selected], width=.34, label=name, color=color)
    ax.axvspan(2.5, 4.5, color=".8", alpha=.18, zorder=0)
    ax.axhline(0, color=".3", linewidth=.8)
    ax.set(xticks=episodes, xlabel="Wall episode (zero-based)", ylabel="Terminal distance regret (Wall units)",
           title="Wall: loss relative to the best endpoint in each captured pool")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=".9", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left")
    fig.text(.5, .01, "Same A-generated candidate pool for both selectors. C-terminal is the zero-regret reference.\n"
             "Episodes 3 and 4 account for 98.64% of B's summed terminal regret; no global-optimality claim.",
             ha="center", fontsize=9, color=".3")
    fig.subplots_adjust(bottom=.23, top=.86)
    save(fig, output, "wall_terminal_regret")
    return [source]


def reach_curves(output):
    source = ROOT / "data/reach_wall/closed_loop/distance_traces.csv"
    data = rows(source)
    traces = {}
    for method in ("A", "B", "C"):
        traces[method] = []
        for ep in range(48):
            group = [r for r in data if r["method"] == method and int(r["episode"]) == ep]
            traces[method].append(dict(episode=ep, steps=np.array([int(r["raw_action"]) for r in group]),
                distance_m=np.array([float(r["tcp_goal_distance_m"]) for r in group]),
                success=np.array([r["native_success"] == "True" for r in group])))
    largest = max(float(t["distance_m"].max() * 100) for group in traces.values() for t in group)
    plot_traces(traces, output, "reach_wall_distance_all", (0, 5 * math.ceil(largest / 5)))
    plot_traces(traces, output, "reach_wall_distance_near_goal", (3, 12), closeup=True)
    fig, ax = plt.subplots(figsize=(10.2, 4.6))
    for method, color in (("A", "#0072B2"), ("B", "#D55E00"), ("C", "#009E73")):
        trace = traces[method][14]
        ax.plot(trace["steps"], trace["distance_m"] * 100, color=color, linewidth=2, label=method)
        ax.scatter(trace["steps"][-1], trace["distance_m"][-1] * 100, color=color, s=45)
    ax.axhline(5, color="#A12A32", linestyle="--", label="Native success: <=5 cm")
    ax.axvline(90, color=".6", linestyle=":", label="Final horizon becomes 10 actions")
    ax.set(xlim=(0, 101), ylim=(0, 35), xlabel="Executed raw simulator actions", ylabel="TCP-to-goal distance (cm)",
           title="Reach-Wall episode 14: actual paired executions")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color=".9")
    fig.tight_layout()
    save(fig, output, "reach_wall_episode_14")
    return [source]


def wall_routes(output):
    source = ROOT / "data/wall/routes/trajectories.npz"
    geometry = ROOT / "data/wall/routes/episode_geometry.csv"
    selected = ROOT / "data/wall/terminal/selections.csv"
    geo = {int(row["episode"]): row for row in rows(geometry)}
    picks = {(int(row["episode"]), row["selection"]): row["route_id"] for row in rows(selected)}
    with np.load(source, allow_pickle=False) as cache:
        fig, axes = plt.subplots(1, 2, figsize=(11, 5.7))
        for ax, ep in zip(axes, (3, 4)):
            indices = np.flatnonzero(cache["episode"] == ep)
            paths = cache["states"][indices]
            lookup = {cache["route_id"][index]: cache["states"][index] for index in indices}
            g = geo[ep]
            for low, high in ((0, 26), (34, 64)):
                ax.add_patch(Rectangle((29, low), 6, high - low, facecolor=".45", zorder=1))
            ax.add_collection(LineCollection(paths, colors=".45", linewidths=.55, alpha=.045, zorder=2))
            goal = np.array([float(g["goal_x"]), float(g["goal_y"])])
            ax.add_patch(Circle(goal, 4.5, facecolor="#F6C344", alpha=.16))
            for label, name, color, linestyle in (("min_terminal", "C-terminal pick", "#009E73", "-"),
                                                  ("min_b", "B pick", "#0072B2", "--")):
                route = lookup[picks[ep, label]]
                ax.plot(route[:, 0], route[:, 1], color=color, linestyle=linestyle, linewidth=2.2, label=name)
                ax.scatter(*route[-1], color=color, s=35, zorder=5)
            ax.scatter(float(g["start_x"]), float(g["start_y"]), color="black", s=28, label="Start", zorder=6)
            ax.scatter(*goal, color="#E5AD29", edgecolors="black", marker="*", s=140, label="Goal", zorder=6)
            ax.set(xlim=(0, 64), ylim=(64, 0), aspect="equal", xlabel="x (Wall units)", ylabel="y (increases downward)",
                   title=f"Episode {ep}: all {len(paths)} captured routes")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
        fig.suptitle("Wall: limited coverage and different terminal selections")
        fig.subplots_adjust(bottom=.15, top=.86, wspace=.25)
        save(fig, output, "wall_candidate_routes")
    return [source, geometry, selected]


def prefix_rounds(output):
    source = ROOT / "data/reach_wall/b_prefix_candidates/cem_round_aggregate.csv"
    data = rows(source)
    x = [int(r["cem_round"]) for r in data]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    axes[0].plot(x, [float(r["sample_success_pct"]) for r in data], marker="o", color="#009E73")
    axes[0].set(ylabel="Successful 15-action sample prefixes (%)", title="Native prefix coverage")
    axes[1].plot(x, [float(r["mean_updated_mean_b_cost"]) for r in data], marker="o", color="#D55E00")
    axes[1].set(ylabel="Mean full-horizon latent cost", title="Optimized endpoint score")
    for ax in axes:
        ax.set(xlabel="CEM round", xticks=(1, 5, 10, 15))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color=".9")
    fig.suptitle("47 audited B replanning states; same original search, no new proposals")
    fig.tight_layout()
    save(fig, output, "prefix_success_by_round")
    return [source]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures/reproduced")
    parser.add_argument("--only", choices=("all", "wall-regret"), default="all")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.size": 10, "pdf.fonttype": 42}):
        sources = wall_regret(output)
        if args.only == "all":
            sources += reach_curves(output) + wall_routes(output) + prefix_rounds(output)
    provenance = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    (output / "manifest.json").write_text(json.dumps(dict(inputs_sha256=provenance, simulator_rerun=False,
        encoder_rerun=False, curve_smoothing=False, curve_padding=False), indent=2) + "\n", encoding="utf-8")
    print(f"Figures saved to {output}")


if __name__ == "__main__":
    main()

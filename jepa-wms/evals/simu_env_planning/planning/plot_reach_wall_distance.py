"""Plot all recorded A/B/C Reach-Wall goal-distance traces without replay.

Run in AI from jepa-wms:
    python -m evals.simu_env_planning.planning.plot_reach_wall_distance

Each line ends at the actual task stopping action. There is no smoothing,
interpolation, padding, or continuation of successful episodes to action 100.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


PROJECT = Path(__file__).resolve().parents[4]
DEFAULT_RUNS = PROJECT / "runs/reach_wall/simu_env_planning"
METHODS = {
    "A": dict(folder="bounded_a_i15_n300_ep48/mw-reach-wall", color="#0072B2", label="Learned dynamics + latent cost"),
    "B": dict(folder="closed_loop_b", color="#D55E00", label="Simulator dynamics + latent cost"),
    "C": dict(folder="closed_loop_c", color="#009E73", label="Simulator dynamics + task criterion"),
}
THRESHOLD_M = .05
EPISODES = 48


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_traces(runs):
    """Read native TCP distance, not A's separate expert-proprio diagnostic."""
    traces, episode_rows, sources = {}, [], {}
    for method, settings in METHODS.items():
        traces[method] = []
        for ep in range(EPISODES):
            path = runs / settings["folder"] / f"ep_{ep}/execution.csv"
            sources[str(path.relative_to(runs))] = digest(path)
            with path.open(newline="", encoding="utf-8-sig") as handle:
                raw = list(csv.DictReader(handle))
            if not raw:
                raise ValueError(f"Empty trajectory: {path}")
            steps = np.array([int(row["simulator_step"]) for row in raw])
            distances = np.array([float(row["tcp_goal_distance_m"]) for row in raw])
            if any(row["success"] not in ("True", "False") for row in raw):
                raise ValueError(f"Invalid native success flag: {path}")
            success = np.array([row["success"] == "True" for row in raw])
            if not np.array_equal(steps, np.arange(1, len(raw)+1)) or len(raw) > 100:
                raise ValueError(f"Noncontiguous or over-budget trajectory: {path}")
            if not np.isfinite(distances).all() or (distances < 0).any():
                raise ValueError(f"Invalid native distance: {path}")
            if not np.array_equal(success, distances <= THRESHOLD_M):
                raise ValueError(f"Success disagrees with native distance threshold: {path}")
            if success[:-1].any() or not success[-1] and len(raw) != 100:
                raise ValueError(f"Episode did not stop at first success or budget: {path}")
            trace = dict(method=method, episode=ep, steps=steps, distance_m=distances, success=success)
            traces[method].append(trace)
            episode_rows.append(dict(method=method, episode=ep, recorded_actions=len(raw), success=bool(success[-1]),
                first_success_action=int(steps[-1]) if success[-1] else None,
                final_distance_cm=float(distances[-1]*100), minimum_distance_cm=float(distances.min()*100)))
    # Validate matched source provenance and completed B/C coverage.
    paired = {}
    for method in ("B", "C"):
        folder = runs / METHODS[method]["folder"]
        coverage = json.loads((folder / "summary.json").read_text())
        if not coverage["complete"] or coverage["completed_episodes"] != EPISODES:
            raise ValueError(f"{method} evaluation is incomplete")
        paired[method] = json.loads((folder / "protocol.json").read_text())
        if paired[method]["source_args_sha256"] != digest(runs / "bounded_a_i15_n300_ep48/args_eval.yaml"):
            raise ValueError(f"{method} uses a different A source config")
        if set(paired[method]["initial_bundles"]) != {str(ep) for ep in range(EPISODES)}:
            raise ValueError(f"{method} is missing paired episode provenance")
    if paired["B"]["initial_bundles"] != paired["C"]["initial_bundles"]:
        raise ValueError("B and C are not paired to the same initial-state bundles")
    return traces, episode_rows, sources


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_traces(traces, output, name, y_limits, closeup=False):
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 5.7), sharex=True, sharey=True)
    for ax, (method, settings) in zip(axes, METHODS.items()):
        records = traces[method]
        color = settings["color"]
        ax.axhspan(0, THRESHOLD_M*100, color="#E8F3E9", alpha=.7, zorder=0)
        for trace in records:
            steps, cm = trace["steps"], trace["distance_m"]*100
            ax.plot(steps, cm, color=color, alpha=.24, linewidth=.9, zorder=2)
            ax.scatter(steps[-1], cm[-1], s=17, marker="o", linewidths=.8,
                       edgecolors=color, facecolors=color if trace["success"][-1] else "white",
                       alpha=.8, zorder=4)
        ax.axhline(THRESHOLD_M*100, color="#A12A32", linestyle="--", linewidth=1.3, zorder=3)
        ax.annotate("5 cm success threshold", xy=(99, 5), xytext=(-3, 5), textcoords="offset points",
                    ha="right", va="bottom", fontsize=8, color="#A12A32")
        successes = sum(bool(trace["success"][-1]) for trace in records)
        ax.set_title(f"{method}: {settings['label']}\n{successes}/{EPISODES} native successes", fontsize=11, pad=12)
        ax.set(xlim=(0, 101), ylim=y_limits, xlabel="Executed raw simulator actions", xticks=[0,20,40,60,80,100])
        ax.grid(color=".89", linewidth=.7, zorder=1)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Actual TCP-to-goal distance (cm)")
    if closeup:
        axes[0].set_yticks([3,4,5,6,7,8,9,10,11,12])
    title = "Reach-Wall: actual goal distance during execution"
    if closeup:
        title += " | near-goal close-up"
    fig.suptitle(title, fontsize=15, y=.98)
    handles = [Line2D([],[],color=".5",alpha=.55,lw=1,label="One recorded episode"),
               Line2D([],[],color=".3",marker="o",linestyle="None",markersize=5,label="Stops at native success"),
               Line2D([],[],color=".3",marker="o",markerfacecolor="white",linestyle="None",markersize=5,label="Stops at action budget")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5,.075), ncol=3, frameon=False, fontsize=9)
    note = "48 paired trials per panel. Each curve ends at success or the 100-action budget."
    note += "\nUnsmoothed distances recorded after each raw action; step 0 was not logged."
    if closeup:
        note += "\nThis close-up shows 3-12 cm; see the full-range figure for the initial approach."
    fig.text(.5,.012,note,ha="center",fontsize=8.5,color=".3")
    fig.subplots_adjust(left=.06,right=.99,top=.82,bottom=.24,wspace=.12)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"{name}.{suffix}",dpi=240,bbox_inches="tight",facecolor="white")
    # Export-level audit: one unmodified finite-length line per actual episode.
    for ax, method in zip(axes, METHODS):
        lines = ax.lines[:-1]  # The last line is the horizontal threshold.
        if len(lines) != EPISODES:
            raise AssertionError("Plot lost or added an episode trace")
        for line, trace in zip(lines, traces[method]):
            np.testing.assert_array_equal(line.get_xdata(),trace["steps"])
            np.testing.assert_array_equal(line.get_ydata(),trace["distance_m"]*100)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir",type=Path,default=DEFAULT_RUNS)
    parser.add_argument("--output-dir",type=Path)
    args=parser.parse_args()
    runs=args.runs_dir.resolve()
    output=(args.output_dir or runs.parent/"analysis/closed_loop_distance").resolve()
    output.mkdir(parents=True,exist_ok=True)
    traces,episode_rows,sources=load_traces(runs)
    largest=max(float(trace["distance_m"].max()*100) for records in traces.values() for trace in records)
    full_limits=(0,5*math.ceil(largest/5))
    with plt.rc_context({"font.size":10,"pdf.fonttype":42}):
        plot_traces(traces,output,"goal_distance_all_episodes",full_limits)
        plot_traces(traces,output,"goal_distance_near_goal",(3,12),closeup=True)
    long_rows=[]
    for method,records in traces.items():
        for trace in records:
            for step,distance,success in zip(trace["steps"],trace["distance_m"],trace["success"]):
                long_rows.append(dict(method=method,episode=trace["episode"],raw_action=int(step),
                                     tcp_goal_distance_m=float(distance),native_success=bool(success)))
    write_csv(output/"distance_traces.csv",long_rows)
    write_csv(output/"episode_summary.csv",episode_rows)
    summary={}
    for method in METHODS:
        rows=[row for row in episode_rows if row["method"]==method]
        summary[method]=dict(episodes=len(rows),successes=sum(row["success"] for row in rows),
                            executed_actions=sum(row["recorded_actions"] for row in rows),
                            mean_final_distance_cm=float(np.mean([row["final_distance_cm"] for row in rows])))
    (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    protocol=dict(source_root=str(runs),source_csv_sha256=sources,native_success_threshold_m=THRESHOLD_M,
                  distance_measurement="Native tcp_goal_distance_m from actual execution, not expert-proprio distance",
                  episodes_per_method=EPISODES,plotted_records=len(long_rows),distance_conversion="cm = metres * 100",
                  source_step_zero_available=False,curves="One line per episode; exact recorded points only; no smoothing/padding/continuation",
                  full_range_y_limits_cm=full_limits,closeup_y_limits_cm=[3,12],
                  caution="C's early stopping is visible; no trajectory values exist beyond its actual stopping point")
    (output/"protocol.json").write_text(json.dumps(protocol,indent=2)+"\n",encoding="utf-8")
    gallery="<!doctype html><meta charset='utf-8'><title>Reach-Wall distance traces</title><style>body{font:17px system-ui;max-width:1550px;margin:30px auto}img{width:100%}section{margin:30px 0}</style><h1>Reach-Wall actual goal-distance traces</h1><p>48 paired episodes per panel. Actual execution only; curves end at task stopping. Click a figure to enlarge.</p>"
    for name,label in (("goal_distance_all_episodes","Full distance range"),("goal_distance_near_goal","Near-goal close-up (3-12 cm)")):
        gallery+=f"<section><h2>{label}</h2><a href='{name}.png'><img src='{name}.png'></a><p><a href='{name}.pdf'>PDF</a></p></section>"
    (output/"index.html").write_text(gallery,encoding="utf-8")
    print(f"Saved 144 episode traces / {len(long_rows):,} actual action outcomes. No replay or model execution.")
    print(json.dumps(summary,indent=2))
    print(f"Figures: {output}")


if __name__=="__main__":
    main()

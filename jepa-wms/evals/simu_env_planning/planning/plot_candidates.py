"""Plot the saved A/B/C analysis without loading a model or simulator.

From jepa-wms, run in the AI environment:
    python -m evals.simu_env_planning.planning.plot_candidates
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Save figures without opening GUI windows.
import matplotlib.pyplot as plt
import numpy as np


WORKSPACE = Path(__file__).resolve().parents[4]
DEFAULT_ANALYSIS = WORKSPACE / "runs/wall/simu_env_planning/pool650_i10/analysis"
METHODS = (
    ("deployed", "A chosen", "#0072B2", "o"),
    ("min_a", "A minimum", "#E69F00", "s"),
    ("min_b", "B minimum", "#009E73", "^"),
    ("best_c", "C best", "#CC79A7", "D"),
)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No data rows in {path}")
    return rows


def is_success(value):
    if value not in ("True", "False"):
        raise ValueError(f"Expected True or False, got {value!r}")
    return value == "True"


def style_axis(ax):
    ax.grid(axis="y", color="0.9", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def save_figure(fig, output_dir, name):
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        path = output_dir / f"{name}.{extension}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved {path}")
    plt.close(fig)


def plot_selection_quality(rows, output_dir):
    rows = sorted(rows, key=lambda row: int(row["episode"]))
    episodes = np.array([int(row["episode"]) for row in rows])
    if len(set(episodes)) != len(episodes):
        raise ValueError("episode_summary.csv contains repeated episode IDs")

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.5), sharex=True)
    offsets = np.linspace(-0.24, 0.24, len(METHODS))

    for offset, (prefix, label, color, marker) in zip(offsets, METHODS):
        success = [is_success(row[f"{prefix}_success"]) for row in rows]
        steps = np.array([
            float(row[f"{prefix}_first_hit_step"]) if ok else np.nan
            for row, ok in zip(rows, success)
        ])
        remaining = np.array([
            np.nan if ok else float(row[f"{prefix}_remaining_distance"])
            for row, ok in zip(rows, success)
        ])
        observed = np.concatenate((steps[success], remaining[np.logical_not(success)]))
        if not np.isfinite(observed).all() or (observed < 0).any():
            raise ValueError(f"Invalid C outcome for {label}")

        axes[0].scatter(
            episodes + offset, steps, color=color, marker=marker, s=58,
            label=f"{label}: {sum(success)}/{len(rows)} success", zorder=3,
        )
        axes[1].scatter(
            episodes + offset, remaining,
            color=color, marker=marker, s=58, zorder=3,
        )
        print(f"{label}: {sum(success)}/{len(rows)} successful selections")

    axes[0].set_title("Successful selections: actions to first success", loc="left")
    axes[0].set_ylabel("Simulator actions (lower is better)")
    axes[0].set_ylim(bottom=0)
    axes[1].set_title("Failed selections: remaining route distance", loc="left")
    axes[1].set_ylabel("Distance units (lower is better)")
    axes[1].set_ylim(bottom=0)
    axes[1].set_xlabel("Episode")
    axes[1].set_xticks(episodes)
    axes[1].set_xlim(episodes.min() - 0.6, episodes.max() + 0.6)
    for ax in axes:
        style_axis(ax)

    fig.suptitle("Selected-route quality under C", fontsize=15, y=0.99)
    fig.legend(
        *axes[0].get_legend_handles_labels(), loc="upper center",
        bbox_to_anchor=(0.5, 0.95), ncol=4, frameon=False, fontsize=9,
    )
    fig.text(
        0.5, 0.015,
        "Each selection appears in one panel per episode. Horizontal offsets separate markers.\n"
        "A/B minima and C best are selected from the shared captured pool.",
        ha="center", fontsize=9, color="0.3",
    )
    fig.tight_layout(rect=(0, 0.065, 1, 0.89), h_pad=2)
    save_figure(fig, output_dir, "selection_quality")


def plot_candidate_quality(rows, output_dir, expected_episodes):
    episodes = sorted({int(row["episode"]) for row in rows})
    rounds = sorted({int(row["cem_round"]) for row in rows})
    lookup = {(int(row["episode"]), int(row["cem_round"])): row for row in rows}
    if len(lookup) != len(rows):
        raise ValueError("round_summary.csv contains repeated episode/round pairs")
    if set(episodes) != set(expected_episodes):
        raise ValueError("Episode IDs differ between the two input summaries")
    if set(lookup) != {(episode, round_) for episode in episodes for round_ in rounds}:
        raise ValueError("Some episode/round pairs are missing")

    rates = np.empty((len(episodes), len(rounds)))
    counts = []
    for i, episode in enumerate(episodes):
        for j, round_ in enumerate(rounds):
            row = lookup[(episode, round_)]
            count, successes = int(row["fresh_count"]), int(row["fresh_success_count"])
            if count <= 0 or not 0 <= successes <= count:
                raise ValueError(f"Invalid fresh-proposal counts: {episode=}, {round_=}")
            rate = successes / count
            if not np.isclose(rate, float(row["fresh_success_rate"]), atol=1e-12, rtol=0):
                raise ValueError("Fresh success rate does not match the saved counts")
            rates[i, j] = 100 * rate
            counts.append(count)

    mean_rates = rates.mean(axis=0)  # Equal weight for each episode.
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for index, episode_rates in enumerate(rates):
        ax.plot(
            rounds, episode_rates, color="0.75", linewidth=1.1, alpha=0.85,
            label="Individual episodes" if index == 0 else None,
        )
    ax.plot(
        rounds, mean_rates, color="#0072B2", marker="o", linewidth=2.5,
        label=f"Mean across {len(episodes)} episodes", zorder=3,
    )
    ax.annotate(
        f"{mean_rates[-1]:.1f}%", (rounds[-1], mean_rates[-1]),
        xytext=(-5, 12), textcoords="offset points", ha="right", color="#0072B2",
    )
    ax.set(
        title="Quality of fresh proposals across CEM rounds", xlabel="CEM round",
        ylabel="Successful fresh proposals (%)", xticks=rounds, ylim=(0, 100),
    )
    style_axis(ax)
    ax.legend(frameon=False, loc="upper left")
    count_note = (
        f"{counts[0]} fresh proposals per episode per round."
        if len(set(counts)) == 1 else "Rates use each episode/round's fresh-proposal count."
    )
    fig.text(
        0.5, 0.02, count_note + " Included and updated means are excluded.\n"
        "Within-round proposal quality; not cumulative or deployed-planner success.",
        ha="center", fontsize=9, color="0.3",
    )
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    save_figure(fig, output_dir, "candidate_quality_by_round")
    print(f"Mean fresh-proposal success: {mean_rates[0]:.1f}% -> {mean_rates[-1]:.1f}%")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    output_dir = args.output_dir or args.analysis_dir / "figures"
    episode_rows = read_csv(args.analysis_dir / "episode_summary.csv")
    round_rows = read_csv(args.analysis_dir / "round_summary.csv")
    with plt.rc_context({"font.size": 10, "pdf.fonttype": 42}):
        plot_selection_quality(episode_rows, output_dir)
        plot_candidate_quality(
            round_rows, output_dir, [int(row["episode"]) for row in episode_rows],
        )


if __name__ == "__main__":
    main()

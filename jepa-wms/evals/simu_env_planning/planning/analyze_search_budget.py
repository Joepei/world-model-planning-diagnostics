"""Extract intermediate CEM means and compare them with their prefix pools.

This is a retrospective analysis of saved ten-round runs, not new planner runs.
Run: python -m evals.simu_env_planning.planning.analyze_search_budget
"""

import argparse
import math
from pathlib import Path
from statistics import mean

from .analyze_candidates import (
    choose_candidates,
    deduplicate_routes,
    group_by,
    load_scores,
    regret_against_pool,
    write_csv,
)


DEFAULT_RUN = (
    Path(__file__).resolve().parents[4]
    / "runs/wall/simu_env_planning/pool650_i10"
)


def extract(rows, budgets):
    records = []
    for episode, episode_rows in sorted(group_by(rows, "episode").items()):
        for budget in budgets:
            means = [
                row for row in episode_rows
                if row["cem_round"] == budget and row["origin"] == "updated_mean"
            ]
            if len(means) != 1:
                raise ValueError(f"Expected one mean for {episode=}, {budget=}")
            selected = means[0]
            prefix = [row for row in episode_rows if row["cem_round"] <= budget]
            unique = deduplicate_routes(prefix)
            oracles = choose_candidates(unique)
            oracle_b, oracle_c = oracles["min_b"], oracles["best_c"]
            records.append({
                "episode": episode,
                "cem_round": budget,
                "candidate": selected["candidate"],
                "is_actual_deployed_output": selected["is_deployed_output"],
                "a_cost": selected["a_cost"],
                "b_cost": selected["b_cost"],
                "b_minus_a": selected["b_cost"] - selected["a_cost"],
                "absolute_a_b_gap": abs(selected["b_cost"] - selected["a_cost"]),
                "c_success": selected["c_success"],
                "c_first_hit_step": selected["c_first_hit_step"],
                "c_remaining_distance": selected["c_remaining_distance"],
                "task_steps": selected["task_steps"],
                "prefix_occurrences": len(prefix),
                "prefix_unique_routes": len(unique),
                "oracle_b_candidate": oracle_b["candidate"],
                "oracle_b_cost": oracle_b["b_cost"],
                "oracle_c_candidate": oracle_c["candidate"],
                "oracle_c_success": oracle_c["c_success"],
                "oracle_c_first_hit_step": oracle_c["c_first_hit_step"],
                "oracle_c_remaining_distance": oracle_c["c_remaining_distance"],
                **regret_against_pool(selected, oracle_b, oracle_c),
            })
    return records


def optional_mean(values):
    return mean(values) if values else None


def summarize(records):
    summaries = []
    for budget, batch in sorted(group_by(records, "cem_round").items()):
        successes = [row for row in batch if row["c_success"]]
        extra_steps = [row["c_excess_steps"] for row in batch if row["c_excess_steps"] is not None]
        extra_distances = [row["c_excess_distance"] for row in batch if row["c_excess_distance"] is not None]
        summaries.append({
            "cem_round": budget,
            "episodes": len(batch),
            "mean_a_cost": mean(row["a_cost"] for row in batch),
            "mean_b_cost": mean(row["b_cost"] for row in batch),
            "mean_b_minus_a": mean(row["b_minus_a"] for row in batch),
            "mean_absolute_a_b_gap": mean(row["absolute_a_b_gap"] for row in batch),
            "mean_b_regret": mean(row["b_regret"] for row in batch),
            "selected_successes": len(successes),
            "prefix_oracle_successes": sum(row["oracle_c_success"] for row in batch),
            "missed_available_successes": sum(row["c_missed_success"] for row in batch),
            "c_optimal_selections": sum(row["c_optimal"] for row in batch),
            "excess_steps_n": len(extra_steps),
            "mean_excess_steps": optional_mean(extra_steps),
            "excess_distance_n": len(extra_distances),
            "mean_excess_distance": optional_mean(extra_distances),
        })
    return summaries


def outcome(row):
    if row["c_success"]:
        return f"S: {row['c_first_hit_step']} actions"
    return f"F: {row['c_remaining_distance']:.2f} distance"


def fmt(value, digits=4):
    return f"{value:.{digits}f}" if value is not None else "n.a."


def write_results(path, records, summaries, source):
    budgets = [row["cem_round"] for row in summaries]
    lines = [
        "# Returned CEM mean by search budget", "",
        f"Source: `{source.resolve().as_posix()}`.", "",
        "Retrospective snapshots of the updated means in the same ten-round runs. "
        "Only round 10 was actually deployed by the original run; earlier means are evaluated saved alternatives. "
        "All means cover the same 30-action horizon. No planner, model, or simulator reruns were performed.", "",
        "Each budget's oracle pool contains only occurrences with cem_round <= budget, "
        "including updated means, deduplicated within the episode. Later candidates are excluded. "
        "A/B are matched full-horizon latent diagnostics; C uses shared stop-on-success task outcomes.", "",
        "## Latent costs of the returned means", "",
        "All quantities below are arithmetic means across the same ten episodes. "
        "B minus A is a signed objective discrepancy; it is not a latent-state prediction error.", "",
        "| Rounds | Mean A | Mean B | Mean B - A | Mean absolute A/B gap | Mean B regret vs prefix pool |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        values = [row[key] for key in ("mean_a_cost", "mean_b_cost", "mean_b_minus_a", "mean_absolute_a_b_gap", "mean_b_regret")]
        lines.append(f"| {row['cem_round']} | " + " | ".join(fmt(value) for value in values) + " |")
    lines += ["", "## Task outcomes and prefix-pool comparisons", "",
        "| Rounds | Selected successes | Pools containing success | Missed available successes | C-optimal selections |",
        "| ---: | ---: | ---: | ---: | ---: |"]
    for row in summaries:
        lines.append(f"| {row['cem_round']} | {row['selected_successes']}/{row['episodes']} | "
                     f"{row['prefix_oracle_successes']}/{row['episodes']} | {row['missed_available_successes']} | {row['c_optimal_selections']} |")
    lines += ["", "| Episode | " + " | ".join(f"Round {budget}" for budget in budgets) + " |",
              "| ---: | " + " | ".join("---:" for _ in budgets) + " |"]
    for episode, batch in sorted(group_by(records, "episode").items()):
        lookup = {row["cem_round"]: row for row in batch}
        lines.append(f"| {episode} | " + " | ".join(outcome(lookup[budget]) for budget in budgets) + " |")
    lines += ["", "S reports first-hit simulator actions; F reports remaining doorway-aware distance at task termination. "
              "Steps and distances are different units and are not combined.", "",
              "## Conditional C regret", "",
              "| Rounds | Mean extra arrival actions (n) | Mean extra remaining distance (n) |",
              "| ---: | ---: | ---: |"]
    for row in summaries:
        lines.append(f"| {row['cem_round']} | {fmt(row['mean_excess_steps'], 2)} ({row['excess_steps_n']}) | "
                     f"{fmt(row['mean_excess_distance'], 2)} ({row['excess_distance_n']}) |")
    lines += ["", "Extra arrival actions apply only when the selected mean succeeds. Extra remaining distance applies "
              "only when no prefix-pool candidate succeeds. A selected failure with an available success is recorded as "
              "a missed success, not a distance or time gap. These conditional averages can cover different episodes at "
              "different budgets; use the per-episode CSV for paired comparisons. C optimum membership respects ties.", "",
              "Regret is relative to the observed prefix pool, not a global physical optimum. "
              "This extraction reports results without assigning a failure mechanism or drawing a research conclusion.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_RUN / "scored_candidates.csv")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--rounds", type=int, nargs="+", default=[1, 2, 5, 10])
    args = parser.parse_args()
    budgets = sorted(set(args.rounds))
    if not budgets or min(budgets) < 1:
        parser.error("Round budgets must be positive integers")
    records = extract(load_scores(args.input), budgets)
    summaries = summarize(records)
    if not all(math.isfinite(row["a_cost"]) and math.isfinite(row["b_cost"]) for row in records):
        raise ValueError("Nonfinite selected costs")
    output = args.output_dir or args.input.parent / "analysis/search_budget"
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "per_episode.csv", records)
    write_csv(output / "budget_summary.csv", summaries)
    write_results(output / "results.md", records, summaries, args.input)
    print((output / "results.md").read_text(encoding="utf-8"))
    print(f"Saved {len(records)} episode/budget rows and {len(summaries)} summaries to {output}")


if __name__ == "__main__":
    main()

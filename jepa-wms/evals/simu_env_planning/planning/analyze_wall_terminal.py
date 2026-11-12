"""Matched-terminal control for the saved Wall A/B/C candidate experiment.

Run in the AI environment from jepa-wms:
    python -m evals.simu_env_planning.planning.analyze_wall_terminal

Only saved scalar scores are read. C-terminal ranks full-horizon Euclidean
goal distance; the original stopped, first-hit C criterion is retained.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median

from scipy.stats import rankdata, spearmanr

from .analyze_candidates import (
    c_key,
    deduplicate_routes,
    group_by,
    load_scores,
    nonnegative_gap,
    number,
    rank_candidates,
    regret_against_pool,
    write_csv,
)


DEFAULT_RUN = Path(__file__).resolve().parents[4] / "runs/wall/simu_env_planning/pool650_i10"
SELECTIONS = {
    "deployed": "A deployed final mean",
    "min_a": "Minimum A",
    "min_b": "Minimum B",
    "min_terminal": "Minimum terminal distance",
    "min_first_hit": "Best original C (first tied representative)",
}
CORRELATIONS = {
    "spearman_b_terminal": ("b_rank", "terminal_rank"),
    "spearman_b_first_hit": ("b_rank", "c_rank"),
    "spearman_terminal_first_hit": ("terminal_rank", "c_rank"),
    "spearman_a_terminal": ("a_rank", "terminal_rank"),
}


def rank_terminal_candidates(rows, threshold=4.5):
    """Rank the same routes under continuous terminal and original C costs."""
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("Goal threshold must be finite and positive")
    if not rows or len({row["episode"] for row in rows}) != 1:
        raise ValueError("Expected a nonempty pool from one episode")
    for row in rows:
        distance = row["end_distance"]
        if not math.isfinite(distance) or distance < 0:
            raise ValueError("Invalid full-horizon terminal distance")
        if distance < threshold and not row["c_success"]:
            raise ValueError("Terminal success must also be a first-hit success")
    ranked = rank_candidates(rows)
    terminal_ranks = rankdata([row["end_distance"] for row in ranked], method="average")
    return [dict(row, terminal_rank=float(rank), terminal_success=row["end_distance"] < threshold)
            for row, rank in zip(ranked, terminal_ranks)]


def correlations(rows):
    result = {}
    for name, (left, right) in CORRELATIONS.items():
        x, y = [row[left] for row in rows], [row[right] for row in rows]
        result[name] = (float(spearmanr(x, y).statistic)
                        if len(set(x)) > 1 and len(set(y)) > 1 else None)
    result["spearman_b_terminal_minus_b_first_hit"] = (
        result["spearman_b_terminal"] - result["spearman_b_first_hit"]
        if result["spearman_b_terminal"] is not None and result["spearman_b_first_hit"] is not None else None
    )
    return result


def analyze_episode(occurrences, threshold=4.5):
    ranked = rank_terminal_candidates(deduplicate_routes(occurrences), threshold)
    lookup = {row["route_id"]: row for row in ranked}
    deployed = [row for row in occurrences if row["is_deployed_output"]]
    if len(deployed) != 1:
        raise ValueError("Expected one actual deployed final mean")
    # Preserve the actual mean even when it has an earlier duplicate alias.
    choices = {
        "deployed": deployed[0],
        "min_a": min(ranked, key=lambda row: row["a_cost"]),
        "min_b": min(ranked, key=lambda row: row["b_cost"]),
        "min_terminal": min(ranked, key=lambda row: row["end_distance"]),
        "min_first_hit": min(ranked, key=c_key),
    }
    best_terminal = choices["min_terminal"]
    best_first_hit = choices["min_first_hit"]
    first_hit_optima = [row for row in ranked if c_key(row) == c_key(best_first_hit)]
    summary = dict(
        episode=ranked[0]["episode"], occurrences=len(occurrences), unique_routes=len(ranked),
        first_hit_success_routes=sum(row["c_success"] for row in ranked),
        terminal_success_routes=sum(row["terminal_success"] for row in ranked),
        first_hit_success_terminal_failure_routes=sum(row["c_success"] and not row["terminal_success"] for row in ranked),
        terminal_optimal_route_count=sum(row["end_distance"] == best_terminal["end_distance"] for row in ranked),
        original_c_optimal_route_count=len(first_hit_optima),
        original_c_optimal_terminal_success_routes=sum(row["terminal_success"] for row in first_hit_optima),
        original_c_optimal_terminal_distance_min=min(row["end_distance"] for row in first_hit_optima),
        original_c_optimal_terminal_distance_max=max(row["end_distance"] for row in first_hit_optima),
        best_terminal_distance=best_terminal["end_distance"],
        **correlations(ranked),
    )
    selections = []
    for name, choice in choices.items():
        rank = lookup[choice["route_id"]]
        old_regret = regret_against_pool(choice, choices["min_b"], best_first_hit)
        selections.append(dict(
            episode=choice["episode"], selection=name, candidate=choice["candidate"],
            route_id=choice["route_id"], cem_round=choice["cem_round"],
            a_cost=choice["a_cost"], b_cost=choice["b_cost"], end_distance=choice["end_distance"],
            terminal_success=rank["terminal_success"], first_hit_success=choice["c_success"],
            first_hit_step=choice["c_first_hit_step"],
            first_hit_remaining_distance=choice["c_remaining_distance"],
            terminal_distance_regret=nonnegative_gap(choice["end_distance"] - best_terminal["end_distance"]),
            terminal_optimal=choice["end_distance"] == best_terminal["end_distance"],
            terminal_missed_success=int(best_terminal["terminal_success"] and not rank["terminal_success"]),
            a_rank=rank["a_rank"], b_rank=rank["b_rank"], terminal_rank=rank["terminal_rank"],
            first_hit_rank=rank["c_rank"], first_hit_optimal=old_regret["c_optimal"],
            first_hit_missed_success=old_regret["c_missed_success"],
            first_hit_excess_steps=old_regret["c_excess_steps"],
            first_hit_excess_remaining_distance=old_regret["c_excess_distance"],
        ))
    return ranked, summary, selections


def descriptive(values):
    values = [value for value in values if value is not None]
    return dict(n=len(values), mean=mean(values) if values else None,
                median=median(values) if values else None,
                minimum=min(values) if values else None, maximum=max(values) if values else None)


def summarize(episodes, selections):
    summary = dict(
        episodes=len(episodes), occurrences=sum(row["occurrences"] for row in episodes),
        unique_routes=sum(row["unique_routes"] for row in episodes),
        pools_with_terminal_success=sum(row["terminal_success_routes"] > 0 for row in episodes),
        pools_with_first_hit_success=sum(row["first_hit_success_routes"] > 0 for row in episodes),
        first_hit_success_routes=sum(row["first_hit_success_routes"] for row in episodes),
        terminal_success_routes=sum(row["terminal_success_routes"] for row in episodes),
        first_hit_success_terminal_failure_routes=sum(row["first_hit_success_terminal_failure_routes"] for row in episodes),
        correlations={}, selections={},
    )
    for field in (*CORRELATIONS, "spearman_b_terminal_minus_b_first_hit"):
        summary["correlations"][field] = descriptive([row[field] for row in episodes])
    for name in SELECTIONS:
        batch = [row for row in selections if row["selection"] == name]
        summary["selections"][name] = dict(
            episodes=len(batch), terminal_successes=sum(row["terminal_success"] for row in batch),
            first_hit_successes=sum(row["first_hit_success"] for row in batch),
            terminal_optimal=sum(row["terminal_optimal"] for row in batch),
            terminal_missed_successes=sum(row["terminal_missed_success"] for row in batch),
            terminal_distance=descriptive([row["end_distance"] for row in batch]),
            terminal_distance_regret=descriptive([row["terminal_distance_regret"] for row in batch]),
        )
    return summary


def write_report(path, source, summary, episodes, selections, threshold, max_steps):
    n = summary["episodes"]
    lines = [
        "# Wall matched-terminal control", "",
        f"Saved source: `{source.resolve().as_posix()}`.", "",
        f"{n} episodes, {summary['occurrences']:,} occurrences, {summary['unique_routes']:,} unique episode/route pairs. "
        "This is a post-hoc analysis; no simulator, encoder, or planner was rerun.", "",
        "## Criteria", "",
        f"- **B:** saved latent goal cost of the actual full-horizon endpoint after {max_steps} raw actions.",
        "- **C-terminal:** saved `end_distance`, the Euclidean position-to-goal distance at that same endpoint. "
        "Smaller distance is better; earlier goal visits and doorway path length do not enter this ranking.",
        f"- **Terminal success:** endpoint distance strictly <{threshold:g}. This is a separate binary diagnostic, not the continuous ranking.",
        "- **Original C / first-hit:** any first-hit success beats failure; earlier arrival wins among successes; "
        "failures rank by remaining doorway-aware geometric distance. Its stopped endpoint can differ from the full-horizon endpoint.", "",
        "Continuation after an earlier success is diagnostic only. The original stop-on-success task remains unchanged. "
        "Distances and regrets below use Wall coordinate units, not metres or action counts.", "",
        "## Within-episode rank agreement", "",
        "Same unique candidates, average ranks for exact ties. Each episode is weighted equally in the mean; "
        "constant-rank correlations are undefined and excluded, with the valid count recorded in summary.json.", "",
        "| Episode | B / terminal | B / original C | Terminal / original C | A / terminal |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    fields = tuple(CORRELATIONS)
    for row in episodes:
        lines.append(f"| {row['episode']} | " + " | ".join(number(row[field]) for field in fields) + " |")
    lines.append("| **Mean** | " + " | ".join(number(summary["correlations"][field]["mean"]) for field in fields) + " |")
    delta = summary["correlations"]["spearman_b_terminal_minus_b_first_hit"]["mean"]
    lines += ["", f"Mean paired difference, B/terminal minus B/original C: {number(delta)}. "
              "This is descriptive; it is not a causal decomposition of disagreement.", "",
              "## Selections over each shared episode pool", "",
              "Terminal regret = selected terminal distance minus the smallest terminal distance in that episode's pool. "
              "The two C selectors optimize different criteria. First occurrence is a representative when minima tie; "
              "all exactly tied terminal minima count as optimal. Original-C optima may have different terminal outcomes; "
              "its displayed representative is not a unique optimum or an independently deployed controller. "
              "The full range across its tied optima is recorded in episode_summary.csv.", "",
              "| Selection | Terminal successes | First-hit successes | Mean terminal distance | Mean terminal regret | Terminal-optimal |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, label in SELECTIONS.items():
        result = summary["selections"][name]
        lines.append(f"| {label} | {result['terminal_successes']}/{n} | {result['first_hit_successes']}/{n} | "
                     f"{number(result['terminal_distance']['mean'])} | {number(result['terminal_distance_regret']['mean'])} | "
                     f"{result['terminal_optimal']}/{n} |")
    lines += ["", "### B's selected route and the terminal reference", "",
              "| Episode | B-selected distance | Best terminal distance | B terminal regret | B terminal success | B first-hit success |",
              "| --- | ---: | ---: | ---: | --- | --- |"]
    for row in selections:
        if row["selection"] != "min_b":
            continue
        best = next(ep["best_terminal_distance"] for ep in episodes if ep["episode"] == row["episode"])
        lines.append(f"| {row['episode']} | {number(row['end_distance'])} | {number(best)} | "
                     f"{number(row['terminal_distance_regret'])} | {row['terminal_success']} | {row['first_hit_success']} |")
    lines += ["", "## First-hit versus terminal outcomes", "",
              f"Pools containing a successful route: first-hit {summary['pools_with_first_hit_success']}/{n}; "
              f"terminal {summary['pools_with_terminal_success']}/{n}.", "",
              f"Across unique routes, {summary['first_hit_success_routes']:,} reach the goal at some point; "
              f"{summary['terminal_success_routes']:,} end inside it; "
              f"{summary['first_hit_success_terminal_failure_routes']:,} reach it but end outside. "
              "These are correlated route records, not independent trials.", "",
              "| Episode | First-hit successes | Terminal successes | Reached goal but ended outside |",
              "| --- | ---: | ---: | ---: |"]
    for row in episodes:
        lines.append(f"| {row['episode']} | {row['first_hit_success_routes']} | {row['terminal_success_routes']} | "
                     f"{row['first_hit_success_terminal_failure_routes']} |")
    lines += ["", "## Interpretation and limits", "",
              "- B versus C-terminal removes the endpoint-versus-first-hit timing mismatch. It tests alignment of the chosen "
              "visual/proprioceptive latent distance and weighting with Euclidean terminal goal distance.",
              "- Disagreement still does not establish that the representation lacks useful state information: "
              "the distance metric, feature scaling, or modality weighting can be unsuitable even when information is present.",
              "- C-terminal versus original C changes both temporal preference and failure-distance geometry "
              "(straight-line versus doorway-aware). A change in correlation cannot be attributed solely to timing. "
              "The reached-then-left count directly describes the temporal outcome distinction.",
              "- Terminal Euclidean distance is not an obstacle-aware cost-to-go; positions near the goal across a wall can score well.",
              "- References are best within the recorded A-generated pool, not global optimal controls. "
              "The earlier unbounded Wall action protocol is unchanged; this analysis makes no bounded-benchmark claim.",
              "- Ten episode cases form a descriptive pilot. Routes and CEM rounds are adaptive and correlated; "
              "no candidate-level significance test is performed.", "",
              "Files: ranked_candidates.csv (unique routes), episode_summary.csv, selections.csv, "
              "summary.json, protocol.json. Original scores and the earlier analysis are preserved.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def analyze(source, output_dir, threshold=4.5, max_steps=30):
    source, output_dir = Path(source), Path(output_dir)
    if source.resolve().parent == output_dir.resolve():
        raise ValueError("Use a separate output directory to preserve original files")
    source_bytes = source.read_bytes()
    rows = load_scores(source, threshold, max_steps)
    ranked, episodes, selections = [], [], []
    for _, batch in sorted(group_by(rows, "episode").items()):
        pool_ranked, episode, selected = analyze_episode(batch, threshold)
        ranked.extend(pool_ranked)
        episodes.append(episode)
        selections.extend(selected)
    if source.read_bytes() != source_bytes:
        raise ValueError("Source scores changed during analysis")
    summary = summarize(episodes, selections)
    protocol = dict(
        source=str(source.resolve()), source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        max_steps=max_steps, terminal_success_threshold=threshold, terminal_success_comparison="strictly less than",
        terminal_ranking="ascending full-horizon Euclidean end_distance; no first-hit or doorway term",
        original_c="success first; earlier first-hit among successes; shorter doorway-aware remaining distance among failures",
        pooling="unique route_id within each episode; first occurrence canonical",
        ranking_ties="average ranks for exactly equal costs; all equal minima optimal",
        selection_ties="first recorded canonical occurrence represents a tied minimum",
        deployed_selection="actual flagged final mean, including its original candidate identity",
        correlation_aggregation="equal episode weights; exclude undefined values, report n",
        rerun_simulator=False, rerun_encoder=False,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, records in (("ranked_candidates.csv", ranked), ("episode_summary.csv", episodes), ("selections.csv", selections)):
        write_csv(output_dir / name, records)
    for name, value in (("summary.json", summary), ("protocol.json", protocol)):
        (output_dir / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_report(output_dir / "results.md", source, summary, episodes, selections, threshold, max_steps)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_RUN / "scored_candidates.csv")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--goal-threshold", type=float, default=4.5)
    parser.add_argument("--max-steps", type=int, default=30)
    args = parser.parse_args()
    output = args.output_dir or args.input.parent / "analysis_terminal"
    result = analyze(args.input, output, args.goal_threshold, args.max_steps)
    print(f"Analyzed {result['occurrences']:,} occurrences / {result['unique_routes']:,} unique routes in {result['episodes']} episodes.")
    for field in ("spearman_b_terminal", "spearman_b_first_hit", "spearman_terminal_first_hit"):
        print(f"{field}: {number(result['correlations'][field]['mean'], 6)}")
    print(f"Report: {output / 'results.md'}")


if __name__ == "__main__":
    main()

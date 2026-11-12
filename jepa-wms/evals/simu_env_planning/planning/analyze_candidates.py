"""Analyze a saved fixed-pool A/B/C experiment without loading a world model.

Run from jepa-wms:
    python -m evals.simu_env_planning.planning.analyze_candidates

A/B describe full-horizon latent diagnostics. C describes the stopped task:
success first, then arrival time; failures are ordered by remaining route length.
"""

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr


FLOAT_FIELDS = ("a_cost", "b_cost", "end_distance", "task_end_distance", "c_remaining_distance")
BOOL_FIELDS = ("c_success", "is_deployed_output", "is_executed_sequence")
INT_FIELDS = ("episode", "cem_round", "task_steps")
OPTIONAL_INTS = ("sample_index", "c_first_hit_step")
ORIGINS = {"gaussian_sample", "included_mean", "updated_mean"}
TOLERANCE = 1e-6


def c_key(row):
    """A smaller tuple is better. Distance does not break successful-route ties."""
    if row["c_success"]:
        return (0, row["c_first_hit_step"])
    return (1, row["c_remaining_distance"])


def parse_bool(value):
    if value not in ("True", "False"):
        raise ValueError(f"Expected True or False, received {value!r}")
    return value == "True"


def load_scores(path, goal_threshold=4.5, max_steps=30):
    """Convert CSV strings to numbers/booleans and check task semantics."""
    required = set(FLOAT_FIELDS + BOOL_FIELDS + INT_FIELDS + OPTIONAL_INTS)
    required.update(("candidate", "origin", "is_elite", "route_id"))
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing CSV fields: {sorted(missing)}")
        rows = []
        for line, raw in enumerate(reader, start=2):
            try:
                row = dict(raw)
                for field in FLOAT_FIELDS:
                    row[field] = float(raw[field])
                    if not math.isfinite(row[field]) or row[field] < 0:
                        raise ValueError(f"Invalid {field}: {raw[field]!r}")
                for field in BOOL_FIELDS:
                    row[field] = parse_bool(raw[field])
                for field in INT_FIELDS:
                    row[field] = int(raw[field])
                for field in OPTIONAL_INTS:
                    row[field] = int(raw[field]) if raw[field] else None
                row["is_elite"] = parse_bool(raw["is_elite"]) if raw["is_elite"] else None
                if row["origin"] not in ORIGINS or not re.fullmatch(r"[0-9a-f]{64}", row["route_id"]):
                    raise ValueError("Invalid origin or route identifier")
                if not 0 <= row["task_steps"] <= max_steps:
                    raise ValueError("Task steps outside the configured budget")
                success = row["c_success"]
                if success != (row["task_end_distance"] < goal_threshold):
                    raise ValueError("Success does not match stopped-task distance")
                if success:
                    if row["c_first_hit_step"] != row["task_steps"]:
                        raise ValueError("Successful task did not stop at its first hit")
                elif row["c_first_hit_step"] is not None or row["task_steps"] != max_steps:
                    raise ValueError("Unsuccessful task has an inconsistent hit or budget")
                if row["c_remaining_distance"] < row["task_end_distance"] - 1e-4:
                    raise ValueError("Route distance is shorter than straight-line distance")
                rows.append(row)
            except (ValueError, TypeError) as error:
                raise ValueError(f"{path.name}, line {line}: {error}") from error
    if not rows:
        raise ValueError("The scored file is empty")
    validate_capture(rows)
    return rows


def group_by(rows, field):
    groups = defaultdict(list)
    for row in rows:
        groups[row[field]].append(row)
    return groups


def validate_capture(rows):
    """Check provenance before interpreting selections or per-round statistics."""
    identities = [(row["episode"], row["candidate"]) for row in rows]
    if len(set(identities)) != len(rows):
        raise ValueError("Repeated episode/candidate occurrence identifiers")
    for episode, episode_rows in group_by(rows, "episode").items():
        rounds = group_by(episode_rows, "cem_round")
        if sorted(rounds) != list(range(1, max(rounds) + 1)):
            raise ValueError(f"Episode {episode}: missing CEM rounds")
        deployed = [row for row in episode_rows if row["is_deployed_output"]]
        if len(deployed) != 1 or deployed[0]["origin"] != "updated_mean" or deployed[0]["cem_round"] != max(rounds):
            raise ValueError(f"Episode {episode}: deployed output must be the final updated mean")
        sizes, elite_counts = set(), set()
        for round_number, batch in rounds.items():
            means = [row for row in batch if row["origin"] == "updated_mean"]
            samples = [row for row in batch if row["origin"] != "updated_mean"]
            if len(means) != 1 or len(samples) < 2:
                raise ValueError(f"Episode {episode}, round {round_number}: invalid pool composition")
            if means[0]["sample_index"] is not None or means[0]["is_elite"] is not None:
                raise ValueError("An updated mean is not a member of the evaluated sample batch")
            if sorted(row["sample_index"] for row in samples) != list(range(len(samples))):
                raise ValueError("Missing or repeated sample indices")
            if any(row["origin"] != ("included_mean" if row["sample_index"] == 0 else "gaussian_sample") for row in samples):
                raise ValueError("Included mean/sample provenance is inconsistent")
            if any(row["is_elite"] is None or row["is_deployed_output"] for row in samples):
                raise ValueError("Invalid sample elite/deployment metadata")
            sizes.add(len(samples))
            elite_counts.add(sum(row["is_elite"] for row in samples))
        if len(sizes) != 1 or len(elite_counts) != 1 or 0 in elite_counts:
            raise ValueError(f"Episode {episode}: sample or elite counts vary between rounds")


def deduplicate_routes(rows):
    """Keep the first occurrence, preserving aliases and auditing duplicate scores."""
    groups = {}
    for row in rows:
        groups.setdefault((row["episode"], row["route_id"]), []).append(row)
    unique = []
    for aliases in groups.values():
        first = aliases[0]
        for other in aliases[1:]:
            if first["c_success"] != other["c_success"] or first["c_first_hit_step"] != other["c_first_hit_step"]:
                raise ValueError(f"Repeated route has different task outcomes: {first['route_id']}")
            for field in ("b_cost", "end_distance", "task_end_distance", "c_remaining_distance"):
                if not math.isclose(first[field], other[field], rel_tol=1e-6, abs_tol=TOLERANCE):
                    raise ValueError(f"Repeated route has different {field}: {first['route_id']}")
        canonical = dict(first)
        canonical.update({
            "occurrence_count": len(aliases),
            "occurrence_candidates": ";".join(row["candidate"] for row in aliases),
            "occurrence_rounds": ";".join(str(row["cem_round"]) for row in aliases),
            "contains_deployed_output": any(row["is_deployed_output"] for row in aliases),
            "repeated_a_cost_span": max(row["a_cost"] for row in aliases) - min(row["a_cost"] for row in aliases),
            "repeated_b_cost_span": max(row["b_cost"] for row in aliases) - min(row["b_cost"] for row in aliases),
        })
        unique.append(canonical)
    return unique


def rank_candidates(rows):
    """Rank aligned rows. Ordinal C codes preserve its tuple ordering and ties."""
    if not rows:
        return []
    keys = [c_key(row) for row in rows]
    ordinal = {key: index for index, key in enumerate(sorted(set(keys)))}
    ranks = {
        "a_rank": rankdata([row["a_cost"] for row in rows], method="average"),
        "b_rank": rankdata([row["b_cost"] for row in rows], method="average"),
        "c_rank": rankdata([ordinal[key] for key in keys], method="average"),
    }
    return [dict(row, **{field: float(values[index]) for field, values in ranks.items()}) for index, row in enumerate(rows)]


def rank_correlations(ranked):
    result = {}
    for left, right in (("a", "b"), ("b", "c"), ("a", "c")):
        x = [row[f"{left}_rank"] for row in ranked]
        y = [row[f"{right}_rank"] for row in ranked]
        # Correlation is undefined for a constant ranking. Do not substitute zero.
        result[f"spearman_{left}{right}"] = (
            float(spearmanr(x, y).statistic) if len(set(x)) > 1 and len(set(y)) > 1 else None
        )
    return result


def choose_candidates(rows):
    return {
        "min_a": min(rows, key=lambda row: row["a_cost"]),
        "min_b": min(rows, key=lambda row: row["b_cost"]),
        "best_c": min(rows, key=c_key),
    }


def nonnegative_gap(value):
    if value < -TOLERANCE:
        raise ValueError(f"Negative oracle gap: {value}")
    return max(0.0, float(value))


def regret_against_pool(choice, oracle_b, oracle_c):
    """Never subtract a geometric distance from an arrival time."""
    result = {
        "b_regret": nonnegative_gap(choice["b_cost"] - oracle_b["b_cost"]),
        "c_optimal": c_key(choice) == c_key(oracle_c),
        "c_missed_success": int(oracle_c["c_success"] and not choice["c_success"]),
        "c_excess_steps": None,
        "c_excess_distance": None,
    }
    if choice["c_success"]:
        if not oracle_c["c_success"]:
            raise ValueError("A successful candidate cannot have an unsuccessful C oracle")
        result["c_excess_steps"] = int(nonnegative_gap(choice["c_first_hit_step"] - oracle_c["c_first_hit_step"]))
    elif not oracle_c["c_success"]:
        result["c_excess_distance"] = nonnegative_gap(choice["c_remaining_distance"] - oracle_c["c_remaining_distance"])
    return result


def outcome_fields(prefix, row, ranks=None):
    result = {
        f"{prefix}_candidate": row["candidate"],
        f"{prefix}_round": row["cem_round"],
        f"{prefix}_success": row["c_success"],
        f"{prefix}_first_hit_step": row["c_first_hit_step"],
        f"{prefix}_remaining_distance": row["c_remaining_distance"],
    }
    if ranks is not None:
        for metric in ("a", "b", "c"):
            result[f"{prefix}_{metric}_rank"] = ranks[f"{metric}_rank"]
    return result


def summarize_episode(rows):
    ranked = rank_candidates(deduplicate_routes(rows))
    lookup = {row["route_id"]: row for row in ranked}
    choices = choose_candidates(ranked)
    # The actual returned mean is identified by provenance, not by lowest A cost.
    choices["deployed"] = next(row for row in rows if row["is_deployed_output"])
    summary = {
        "episode": rows[0]["episode"],
        "occurrences": len(rows),
        "unique_routes": len(ranked),
        "successful_unique_routes": sum(row["c_success"] for row in ranked),
        "c_optimal_route_count": sum(c_key(row) == c_key(choices["best_c"]) for row in ranked),
        "max_repeated_a_cost_span": max(row["repeated_a_cost_span"] for row in ranked),
        "max_repeated_b_cost_span": max(row["repeated_b_cost_span"] for row in ranked),
        **rank_correlations(ranked),
    }
    regrets = {"episode": rows[0]["episode"]}
    for name in ("deployed", "min_a", "min_b", "best_c"):
        choice = choices[name]
        summary.update(outcome_fields(name, choice, lookup[choice["route_id"]]))
        summary[f"{name}_c_optimal"] = c_key(choice) == c_key(choices["best_c"])
        if name != "best_c":
            regrets[f"{name}_candidate"] = choice["candidate"]
            regrets.update({f"{name}_{key}": value for key, value in regret_against_pool(choice, choices["min_b"], choices["best_c"]).items()})
    regrets["deployed_a_return_gap"] = nonnegative_gap(choices["deployed"]["a_cost"] - choices["min_a"]["a_cost"])
    regrets.update(outcome_fields("oracle_c", choices["best_c"]))
    return ranked, summary, regrets


def optional_median(values):
    return float(np.median(values)) if values else None


def summarize_round(rows, cumulative_rows):
    ranked = rank_candidates(rows)  # All current-round occurrences are retained.
    choices = choose_candidates(ranked)
    fresh = [row for row in rows if row["origin"] == "gaussian_sample"]
    samples = [row for row in rows if row["origin"] != "updated_mean"]
    elites = [row for row in samples if row["is_elite"]]
    mean = next(row for row in rows if row["origin"] == "updated_mean")
    successes = [row for row in fresh if row["c_success"]]
    failures = [row for row in fresh if not row["c_success"]]
    summary = {
        "episode": rows[0]["episode"],
        "cem_round": rows[0]["cem_round"],
        "round_occurrences": len(rows),
        "fresh_count": len(fresh),
        "fresh_success_count": len(successes),
        "fresh_success_rate": len(successes) / len(fresh),
        "fresh_success_first_hit_median": optional_median([row["c_first_hit_step"] for row in successes]),
        "fresh_failure_remaining_median": optional_median([row["c_remaining_distance"] for row in failures]),
        "fresh_failure_remaining_q25": float(np.quantile([row["c_remaining_distance"] for row in failures], 0.25)) if failures else None,
        "fresh_failure_remaining_q75": float(np.quantile([row["c_remaining_distance"] for row in failures], 0.75)) if failures else None,
        "elite_success_rate": sum(row["c_success"] for row in elites) / len(elites),
        **rank_correlations(ranked),
    }
    for name, choice in choices.items():
        summary.update(outcome_fields(name, choice))
        summary[f"{name}_c_optimal"] = c_key(choice) == c_key(choices["best_c"])
    summary.update(outcome_fields("updated_mean", mean))
    summary.update(outcome_fields("best_sample_c", min(samples, key=c_key)))
    summary.update(outcome_fields("cumulative_best_c", min(cumulative_rows, key=c_key)))
    return summary


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def number(value, digits=3):
    return f"{value:.{digits}f}" if value is not None else "n.a."


def write_report(path, source, ranked, episodes, rounds, regrets, threshold, max_steps):
    n = len(episodes)
    lines = [
        "# Fixed-pool A/B/C analysis", "",
        f"Source: `{source.resolve().as_posix()}`.", "",
        f"{n} episodes; {sum(row['occurrences'] for row in episodes):,} occurrence records; "
        f"{len(ranked):,} unique episode/route pairs. The first recorded occurrence is canonical.", "",
        "## Protocol and interpretation", "",
        f"Task execution stops at the first actual goal distance <{threshold:g}, or after {max_steps} actions. "
        "A and B retain matched full-horizon latent diagnostics. Counterfactual continuation after success is excluded from task action counts.", "",
        "C ranks success before failure, then earlier first-hit step for successes and shorter remaining doorway-aware geometric distance for failures. "
        "Ranks use average positions for ties. Different successful routes with the same first-hit step are equally good under C.", "",
        "Pooled analyses deduplicate within episodes. Round summaries retain occurrences and identify fresh Gaussian proposals separately. "
        "Candidates are adaptive and correlated; the ten episode cases, not the thousands of route records, are the units of the pilot. No candidate-level significance tests are reported.", "",
        "## Pooled rank agreement", "",
        "Spearman correlations compare aligned ranks for the same unique routes within each episode. "
        "An undefined constant-rank correlation is reported as n.a., not zero. "
        "[Method reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html).", "",
        "| Episode | A/B | B/C | A/C | Unique routes |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in episodes:
        lines.append(f"| {row['episode']} | {number(row['spearman_ab'])} | {number(row['spearman_bc'])} | {number(row['spearman_ac'])} | {row['unique_routes']} |")
    lines += ["", "| Pair | Mean across episodes | Median | Minimum | Maximum |", "| --- | ---: | ---: | ---: | ---: |"]
    for pair in ("ab", "bc", "ac"):
        values = [row[f"spearman_{pair}"] for row in episodes if row[f"spearman_{pair}"] is not None]
        stats = [float(np.mean(values)), float(np.median(values)), min(values), max(values)] if values else [None] * 4
        lines.append(f"| {pair.upper()} | " + " | ".join(number(value) for value in stats) + " |")
    lines += ["", "## Selected task outcomes", "", "| Selection rule | Successes | C-optimal selections (ties respected) |", "| --- | ---: | ---: |"]
    for prefix, label in (("deployed", "Deployed final mean"), ("min_a", "Minimum A in unique pool"), ("min_b", "Minimum B in unique pool"), ("best_c", "Best C in unique pool")):
        lines.append(f"| {label} | {sum(row[f'{prefix}_success'] for row in episodes)}/{n} | {sum(row[f'{prefix}_c_optimal'] for row in episodes)}/{n} |")
    missing = [str(row["episode"]) for row in episodes if not row["best_c_success"]]
    lines += ["", f"Pools without any successful route: {', '.join(missing) if missing else 'none'}. This is a limitation of the recorded pool, not proof that the task is unsolvable.", "",
        "## Deployed planning regret", "",
        "B regret is B(deployed) minus the best pooled B cost. C regret reports missed success, excess arrival steps when the deployed route succeeds, "
        "or excess remaining distance when no pooled route succeeds. Blank/inapplicable components are n.a.; distances and steps are never added.", "",
        "| Episode | B regret | Missed success | Extra steps | Extra remaining distance |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in regrets:
        lines.append(f"| {row['episode']} | {number(row['deployed_b_regret'], 6)} | {row['deployed_c_missed_success']} | {number(row['deployed_c_excess_steps'], 0)} | {number(row['deployed_c_excess_distance'])} |")
    lines += ["", "`planning_regret.csv` also evaluates minimum-A and minimum-B choices against the same oracles. "
        "The deployed mean is not redefined as the minimum-A route. These are regrets relative to the observed pool, not global optimal-control regret.", "",
        "## Candidate quality across rounds", "",
        "Fresh-proposal success rates are averaged equally across episodes. Correlations below are within-round correlations averaged over episodes. "
        "The saved round file also contains conditional arrival-step/failure-distance distributions, elite outcomes, mean outcomes, and current/cumulative best C outcomes.", "",
        "| Round | Fresh success rate | Updated-mean successes | Mean A/B rho | Mean B/C rho | Mean A/C rho |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    by_round = group_by(rounds, "cem_round")
    for k in sorted(by_round):
        batch = by_round[k]
        correlations = []
        for pair in ("ab", "bc", "ac"):
            values = [row[f"spearman_{pair}"] for row in batch if row[f"spearman_{pair}"] is not None]
            correlations.append(float(np.mean(values)) if values else None)
        lines.append(f"| {k} | {np.mean([row['fresh_success_rate'] for row in batch]):.1%} | {sum(row['updated_mean_success'] for row in batch)}/{len(batch)} | " + " | ".join(number(value) for value in correlations) + " |")
    first, last = min(by_round), max(by_round)
    first_rates = {row["episode"]: row["fresh_success_rate"] for row in by_round[first]}
    changes = [row["fresh_success_rate"] - first_rates[row["episode"]] for row in by_round[last]]
    lines += ["", f"Fresh success rate, last minus first round: mean paired change {100 * np.mean(changes):+.1f} percentage points; "
        f"{sum(value > 0 for value in changes)} episodes improve, {sum(value == 0 for value in changes)} tie, {sum(value < 0 for value in changes)} decrease.", "",
        "Cumulative best quality is monotonic by construction, so it cannot alone show that later proposals improve. "
        "Conditional step and distance summaries describe changing subpopulations; success rate must be read alongside them.", "",
        "## Limits and audit", "",
        "- A/B disagreement concerns prediction effects on cost ordering; equal A/B cost does not prove all predicted features are correct.",
        "- B/C disagreement includes both latent geometry and the deliberate endpoint-versus-first-hit distinction. It is not automatically an encoder failure.",
        "- C tie-breaking is only for displaying a representative. All C-optimal tied routes are counted as optimal.",
        f"- Maximum repeated-route A-cost span: {max(row['max_repeated_a_cost_span'] for row in episodes):.9g}. "
        f"Maximum B-cost span: {max(row['max_repeated_b_cost_span'] for row in episodes):.9g}. "
        "B/C duplicate consistency and capture/task metadata were checked before analysis.",
        "- The existing ten episodes constitute a pilot; conclusions are limited to these cases and captured routes.", "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def analyze(source, output_dir, goal_threshold=4.5, max_steps=30):
    rows = load_scores(source, goal_threshold, max_steps)
    all_ranked, episode_summaries, round_summaries, regrets = [], [], [], []
    for _, episode_rows in sorted(group_by(rows, "episode").items()):
        ranked, summary, regret = summarize_episode(episode_rows)
        all_ranked.extend(ranked)
        episode_summaries.append(summary)
        regrets.append(regret)
        cumulative = []
        for _, batch in sorted(group_by(episode_rows, "cem_round").items()):
            cumulative.extend(batch)
            round_summaries.append(summarize_round(batch, cumulative))
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, records in (("ranked_candidates.csv", all_ranked), ("episode_summary.csv", episode_summaries),
                              ("round_summary.csv", round_summaries), ("planning_regret.csv", regrets)):
        write_csv(output_dir / filename, records)
    write_report(output_dir / "analysis_summary.md", source, all_ranked, episode_summaries, round_summaries, regrets, goal_threshold, max_steps)
    return {"occurrences": len(rows), "unique_routes": len(all_ranked), "episodes": len(episode_summaries), "rounds": len(round_summaries)}


def main():
    default_run = Path(__file__).resolve().parents[4] / "runs/wall/simu_env_planning/pool650_i10"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=default_run / "scored_candidates.csv")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--goal-threshold", type=float, default=4.5)
    parser.add_argument("--max-steps", type=int, default=30)
    args = parser.parse_args()
    output_dir = args.output_dir or args.input.parent / "analysis"
    counts = analyze(args.input, output_dir, args.goal_threshold, args.max_steps)
    print(f"Analyzed {counts['occurrences']:,} entries / {counts['unique_routes']:,} unique routes across {counts['episodes']} episodes.")
    print(f"Saved pooled rankings, {counts['rounds']} round summaries, regret, and the report to:\n{output_dir}")


if __name__ == "__main__":
    main()

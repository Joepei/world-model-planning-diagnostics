"""Matched-terminal C control for Reach-Wall's saved A-generated shared pools.

Reads the completed B database only; no simulator, encoder, or controller runs.
C-terminal is continuous native TCP-to-goal distance at the diagnostic endpoint.
The original first-hit analysis and actual stop-on-success task are preserved.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sqlite3

import numpy as np
from scipy.stats import rankdata, spearmanr

from .analyze_candidates import c_key, deduplicate_routes, rank_candidates, write_csv
from .reach_wall_c import DEFAULT_RUN, c_outcome, expected_pools


SCOPES = ("full_horizon", "executed_prefix")
LABELS = {
    "a_deployed_mean": "A deployed final mean",
    "a_pool_min": "Minimum A cost",
    "b_pool_min": "Minimum B cost",
    "c_terminal_best": "Minimum terminal C distance",
    "c_first_hit_best": "Original first-hit C (tied representative)",
}
CORRELATIONS = {
    "ab": ("a_rank", "b_rank"),
    "a_terminal": ("a_rank", "terminal_rank"),
    "b_terminal": ("b_rank", "terminal_rank"),
    "a_first_hit": ("a_rank", "c_rank"),
    "b_first_hit": ("b_rank", "c_rank"),
    "terminal_first_hit": ("terminal_rank", "c_rank"),
}


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def descriptive(values):
    values = [value for value in values if value is not None]
    return dict(n=len(values), mean=float(np.mean(values)) if values else None,
                median=float(np.median(values)) if values else None)


def analyze_pool(occurrences, scope):
    rows = [c_outcome(row, scope) for row in occurrences]
    if not rows or len({(row["episode"], row["plan_call"]) for row in rows}) != 1:
        raise ValueError("Expected a nonempty pool from one decision state")
    if sorted(row["candidate_index"] for row in rows) != list(range(len(rows))):
        raise ValueError("Incomplete candidate occurrence indices")
    endpoint_field = "full_horizon_goal_distance_m" if scope == "full_horizon" else "prefix_horizon_goal_distance_m"
    for row in rows:
        distance = row[endpoint_field]
        if not math.isfinite(distance) or distance < 0:
            raise ValueError("Invalid native terminal distance")
        row["terminal_distance_m"] = distance
        row["terminal_success"] = distance <= .05
        if row["terminal_success"] and not row["c_success"]:
            raise ValueError("Terminal success must imply a first-hit success")
        if scope == "full_horizon" and bool(row["full_horizon_success"]) != row["terminal_success"]:
            raise ValueError("Native full-horizon success and endpoint distance disagree")
    ranked = rank_candidates(deduplicate_routes(rows))
    terminal_ranks = rankdata([row["terminal_distance_m"] for row in ranked], method="average")
    for row, rank in zip(ranked, terminal_ranks):
        row["terminal_rank"] = float(rank)
    lookup = {row["route_id"]: row for row in ranked}
    # Include prefix endpoints in the duplicate audit, not just the full endpoint.
    for row in rows:
        canonical = lookup[row["route_id"]]
        if not math.isclose(row["terminal_distance_m"], canonical["terminal_distance_m"], abs_tol=1e-6, rel_tol=1e-6):
            raise ValueError("Duplicate route has inconsistent terminal distance")
    deployed = [row for row in rows if row["is_deployed_output"]]
    if len(deployed) != 1:
        raise ValueError("Expected exactly one A deployed mean")
    choices = dict(
        a_deployed_mean=deployed[0], a_pool_min=min(rows, key=lambda row: row["a_cost"]),
        b_pool_min=min(rows, key=lambda row: row["b_cost"]),
        c_terminal_best=min(ranked, key=lambda row: row["terminal_distance_m"]),
        c_first_hit_best=min(ranked, key=c_key),
    )
    first, best = rows[0], choices["c_terminal_best"]
    base = dict(episode=first["episode"], plan_call=first["plan_call"], scope=scope,
                episode_steps_elapsed=first["episode_steps_elapsed"],
                evaluation_horizon=first["full_horizon_steps"] if scope == "full_horizon" else first["prefix_steps"])
    pool = dict(base, occurrences=len(rows), unique_routes=len(ranked),
                first_hit_success_routes=sum(row["c_success"] for row in ranked),
                terminal_success_routes=sum(row["terminal_success"] for row in ranked),
                reached_then_left_routes=sum(row["c_success"] and not row["terminal_success"] for row in ranked),
                best_terminal_distance_m=best["terminal_distance_m"])
    for name, (left, right) in CORRELATIONS.items():
        x, y = [row[left] for row in ranked], [row[right] for row in ranked]
        pool["spearman_" + name] = float(spearmanr(x, y).statistic) if len(set(x)) > 1 and len(set(y)) > 1 else None
    for target in ("a_deployed_mean", "a_pool_min"):
        b, a = choices["b_pool_min"]["terminal_distance_m"], choices[target]["terminal_distance_m"]
        pool["b_vs_" + target] = "better" if b < a else "worse" if b > a else "tied"
        b, a = c_key(choices["b_pool_min"]), c_key(choices[target])
        pool["first_hit_b_vs_" + target] = "better" if b < a else "worse" if b > a else "tied"
    selections = []
    for label, choice in choices.items():
        rank = lookup[choice["route_id"]]
        regret = choice["terminal_distance_m"] - best["terminal_distance_m"]
        if regret < -1e-6:
            raise ValueError("Negative terminal regret")
        selections.append(dict(
            base, selection=label, candidate=choice["candidate"], candidate_index=choice["candidate_index"],
            route_id=choice["route_id"], cem_round=choice["cem_round"],
            a_cost=choice["a_cost"], b_cost=choice["b_cost"],
            terminal_distance_m=choice["terminal_distance_m"], terminal_success=choice["terminal_success"],
            first_hit_success=choice["c_success"], first_hit_step=choice["c_first_hit_step"],
            terminal_regret_m=max(0., regret), terminal_optimal=regret == 0,
            terminal_missed_success=int(best["terminal_success"] and not choice["terminal_success"]),
            a_rank=rank["a_rank"], b_rank=rank["b_rank"], terminal_rank=rank["terminal_rank"], first_hit_rank=rank["c_rank"],
        ))
    return ranked, pool, selections


def aggregate(pools, selections):
    result = dict(completed_pools=len(pools) // len(SCOPES), scopes={})
    for scope in SCOPES:
        batch = [row for row in pools if row["scope"] == scope]
        summary = dict(pools=len(batch), occurrences=sum(row["occurrences"] for row in batch),
                       unique_routes=sum(row["unique_routes"] for row in batch),
                       pools_with_first_hit_success=sum(row["first_hit_success_routes"] > 0 for row in batch),
                       pools_with_terminal_success=sum(row["terminal_success_routes"] > 0 for row in batch),
                       first_hit_success_routes=sum(row["first_hit_success_routes"] for row in batch),
                       terminal_success_routes=sum(row["terminal_success_routes"] for row in batch),
                       reached_then_left_routes=sum(row["reached_then_left_routes"] for row in batch),
                       correlations={}, selections={}, terminal_comparisons={}, first_hit_comparisons={})
        for name in CORRELATIONS:
            summary["correlations"][name] = descriptive([row["spearman_" + name] for row in batch])
        for label in LABELS:
            selected = [row for row in selections if row["scope"] == scope and row["selection"] == label]
            summary["selections"][label] = dict(
                terminal_successes=sum(row["terminal_success"] for row in selected),
                first_hit_successes=sum(row["first_hit_success"] for row in selected),
                terminal_optimal=sum(row["terminal_optimal"] for row in selected),
                terminal_missed_successes=sum(row["terminal_missed_success"] for row in selected),
                terminal_distance_m=descriptive([row["terminal_distance_m"] for row in selected]),
                terminal_regret_m=descriptive([row["terminal_regret_m"] for row in selected]),
            )
        for target in ("a_deployed_mean", "a_pool_min"):
            for name, prefix in (("terminal_comparisons", ""), ("first_hit_comparisons", "first_hit_")):
                summary[name]["b_vs_" + target] = {key: sum(row[prefix + "b_vs_" + target] == key for row in batch)
                                                    for key in ("better", "tied", "worse")}
        result["scopes"][scope] = summary
    return result


def write_report(output, summary):
    lines = ["# Reach-Wall shared-pool matched-terminal control", "",
             f"Complete: {summary['completed_pools']}/336 decision pools from 48 A episodes. No simulator, encoder, or planner rerun.", "",
             "C-terminal ranks continuous native TCP-to-goal distance at the full diagnostic endpoint, lower is better. "
             "Terminal success is separately defined as distance <=0.05 m. Earlier goal visits and arrival times do not affect this ranking. "
             "A/B retain their saved full-horizon latent scores and original candidate selections.", "",
             "The primary full_horizon control matches A/B's scoring time. The executed_prefix view ranks distances at the prefix endpoint "
             "(usually action 15), while A/B still select by full-horizon cost: it is an execution diagnostic, not a matched-time latent comparison.", ""]
    for scope, data in summary["scopes"].items():
        n = data["pools"]
        lines += ["## " + scope, "",
                  f"{data['occurrences']:,} occurrences, {data['unique_routes']:,} unique decision/route pairs. "
                  f"Pools with success: first-hit {data['pools_with_first_hit_success']}/{n}; terminal {data['pools_with_terminal_success']}/{n}.", "",
                  "| Selection | First-hit successes | Terminal successes | Mean terminal distance (cm) | Mean terminal regret (cm) |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        for label, name in LABELS.items():
            row = data["selections"][label]
            lines.append(f"| {name} | {row['first_hit_successes']}/{n} | {row['terminal_successes']}/{n} | "
                         f"{100 * row['terminal_distance_m']['mean']:.4f} | {100 * row['terminal_regret_m']['mean']:.4f} |")
        lines += ["", "Terminal regret is selected endpoint distance minus the best distance in that decision's recorded pool. "
                  "Original first-hit C has tied optima; its displayed representative is not uniquely preferred and can have a different terminal distance from other tied optima.", "",
                  "| Rank comparison | Mean within-pool Spearman | Valid pools |", "| --- | ---: | ---: |"]
        for name, row in data["correlations"].items():
            value = f"{row['mean']:.6f}" if row["mean"] is not None else "undefined"
            lines.append(f"| {name} | {value} | {row['n']} |")
        for name, comparisons in (("Terminal C", data["terminal_comparisons"]), ("Original first-hit C", data["first_hit_comparisons"])):
            row = comparisons["b_vs_a_deployed_mean"]
            lines += ["", f"{name}: B beats A's deployed mean in {row['better']} pools, ties in {row['tied']}, loses in {row['worse']}."]
        lines += ["", f"Across unique routes: {data['first_hit_success_routes']:,} reach the goal within the horizon; "
                  f"{data['terminal_success_routes']:,} end inside it; {data['reached_then_left_routes']:,} reach it and end outside.", ""]
    lines += ["## Scope", "",
              "- Candidates and states all come from A. These are conditional route selections, not independently deployed controller episode success rates.",
              "- Distance is a native measurement in metres and a Euclidean progress proxy, not obstacle-aware cost-to-go or a global optimal-control reference.",
              "- Full-horizon B versus C-terminal removes the first-hit versus endpoint timing mismatch. Remaining disagreement concerns alignment of the chosen latent metric, modality weights and goal with native TCP distance; it does not establish missing information in the encoder.",
              "- Original first-hit C versus C-terminal also changes preferences among successful routes: earliest hit becomes smallest endpoint distance. The difference is not solely whether reached-then-left routes count as successes.",
              "- Exact action duplicates are collapsed within each decision; exact score ties receive average ranks. Pool means weight each decision equally. Seven decisions per episode are related; no candidate-level significance test is used.",
              "- Post-success continuation is diagnostic only. Actual episode stopping and the original first-hit outputs remain unchanged.", "",
              "Reproduce from jepa-wms: `python -m evals.simu_env_planning.planning.analyze_reach_wall_terminal`.", ""]
    (output / "results.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--write-ranked", action="store_true", help="Also export all unique-route ranks (large CSV)")
    args = parser.parse_args(argv)
    run = args.run_dir.resolve()
    output = (args.output_dir or run / "branch_c_terminal").resolve()
    if output in {run / "branch_b", run / "branch_c", run / "branch_c_partial"}:
        raise ValueError("Use a separate terminal-analysis output directory")
    sources = [run / "branch_b/scores.sqlite3", run / "args_eval.yaml", run / "branch_c/summary.json",
               run / "branch_c/selections.csv", Path(__file__), Path(__file__).with_name("reach_wall_c.py"),
               Path(__file__).with_name("analyze_candidates.py")]
    hashes = {str(path): sha256(path) for path in sources}
    expected = expected_pools(run)
    connection = sqlite3.connect((run / "branch_b/scores.sqlite3").as_uri() + "?mode=ro", uri=True)
    connection.execute("BEGIN")
    counts = connection.execute("SELECT p.episode,p.plan_call,p.total,count(s.candidate_index) FROM pools p LEFT JOIN scores s ON p.episode=s.episode AND p.plan_call=s.plan_call GROUP BY p.episode,p.plan_call ORDER BY p.episode,p.plan_call").fetchall()
    if {(ep, call) for ep, call, _, _ in counts} != expected or any(total != count for _, _, total, count in counts):
        connection.close()
        raise ValueError("Complete B coverage is required")
    output.mkdir(parents=True, exist_ok=True)
    pools, selections = [], []
    ranked_handle = (output / "ranked_candidates.csv").open("w", newline="", encoding="utf-8") if args.write_ranked else None
    writer = None
    try:
        if ranked_handle:
            import csv
        for index, (ep, call, total, _) in enumerate(counts):
            raw = [json.loads(payload) for (payload,) in connection.execute(
                "SELECT payload FROM scores WHERE episode=? AND plan_call=? ORDER BY candidate_index", (ep, call))]
            if len(raw) != total:
                raise ValueError("Incomplete pool")
            for scope in SCOPES:
                ranked, pool, selected = analyze_pool(raw, scope)
                pools.append(pool)
                selections.extend(selected)
                if ranked_handle:
                    fields = ("episode", "plan_call", "candidate", "route_id", "a_cost", "b_cost", "terminal_distance_m",
                              "terminal_success", "c_success", "c_first_hit_step", "a_rank", "b_rank", "terminal_rank", "c_rank")
                    compact = [dict(scope=scope, **{key: row[key] for key in fields}) for row in ranked]
                    if writer is None:
                        writer = csv.DictWriter(ranked_handle, fieldnames=list(compact[0]))
                        writer.writeheader()
                    writer.writerows(compact)
            if index % 20 == 0 or index + 1 == len(counts):
                print(f"Terminal analysis: {index + 1}/{len(counts)} pools", flush=True)
    finally:
        connection.close()
        if ranked_handle:
            ranked_handle.close()
    summary = aggregate(pools, selections)
    summary.update(expected_pools=len(expected), complete=len(counts) == len(expected), episodes=len({ep for ep, _ in expected}))
    # Confirm that only C's criterion changed: the original A/B choices and
    # first-hit correlations/comparisons must reproduce the completed analysis.
    original = json.loads((run / "branch_c/summary.json").read_text())
    for scope in SCOPES:
        old, new = original["scopes"][scope], summary["scopes"][scope]
        if new["pools_with_first_hit_success"] != old["pools_with_success"] or new["first_hit_comparisons"] != old["comparisons"]:
            raise ValueError("Original first-hit coverage/comparisons did not reproduce")
        for label in ("a_deployed_mean", "a_pool_min", "b_pool_min"):
            if new["selections"][label]["first_hit_successes"] != old["selections"][label]["successes"]:
                raise ValueError("Original A/B first-hit outcomes did not reproduce")
        for new_name, old_name in (("ab", "ab"), ("a_first_hit", "ac"), ("b_first_hit", "bc")):
            if not math.isclose(new["correlations"][new_name]["mean"], old["mean_spearman"][old_name], abs_tol=1e-12):
                raise ValueError("Original rank correlations did not reproduce")
    if any(sha256(Path(path)) != value for path, value in hashes.items()):
        raise ValueError("Source files changed during analysis")
    protocol = dict(source_sha256=hashes, complete=True, expected_pools=len(expected),
                    c_terminal="ascending native diagnostic endpoint TCP-to-goal distance in metres for all candidates",
                    terminal_success="distance <=0.05 m; separate binary diagnostic", no_first_hit_tiebreak=True,
                    primary_scope="full_horizon", prefix_scope_matches_latent_scoring_time=False,
                    rerun_simulator=False, rerun_encoder=False, original_first_hit_reproduced=True)
    for name, rows in (("pool_summary.csv", pools), ("selections.csv", selections),
                       ("planning_regret.csv", [row for row in selections if row["selection"] in ("a_deployed_mean", "a_pool_min", "b_pool_min")])):
        write_csv(output / name, rows)
    for name, value in (("summary.json", summary), ("protocol.json", protocol)):
        (output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_report(output, summary)
    print(json.dumps(summary, indent=2), flush=True)
    print("Results:", output / "results.md", flush=True)
    return summary


if __name__ == "__main__":
    main()

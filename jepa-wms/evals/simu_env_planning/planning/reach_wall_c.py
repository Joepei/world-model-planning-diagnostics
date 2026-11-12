"""Ground-truth C ordering and decision-local A/B/C analysis from saved B data.

No simulator/model is loaded. C is lexicographic: success, then first-hit action;
failures rank by stopped endpoint TCP distance. It is a shared-pool reference,
not a globally optimal or obstacle-aware motion planner.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import sqlite3

import numpy as np
import yaml

from evals.simu_env_planning.planning.analyze_candidates import (
    c_key, deduplicate_routes, group_by, rank_candidates, rank_correlations,
    regret_against_pool, summarize_round, write_csv,
)

DEFAULT_RUN = Path(__file__).resolve().parents[4] / "runs/reach_wall/simu_env_planning/bounded_a_i15_n300_ep48"
LABELS = ("a_deployed_mean", "a_pool_min", "b_pool_min", "c_pool_best")


def c_outcome(row, scope="full_horizon"):
    """Map native measurements to C without combining steps and metres."""
    if scope not in ("full_horizon", "executed_prefix"):
        raise ValueError("Unknown C scope")
    prefix = scope == "executed_prefix"
    success = bool(row["prefix_success"] if prefix else row["task_success"])
    hit = row["task_first_hit_step"] if success else None
    steps = row["prefix_task_steps"] if prefix else row["task_steps"]
    horizon = row["prefix_steps"] if prefix else row["full_horizon_steps"]
    distance = row["prefix_task_end_distance_m"] if prefix else row["task_end_distance_m"]
    if not math.isfinite(distance) or distance < 0 or not 0 <= steps <= horizon:
        raise ValueError("Invalid native C outcome")
    if success != (distance <= .05):
        raise ValueError("Native success and stopped TCP distance disagree")
    if success and (hit != steps or hit is None) or not success and (hit is not None or steps != horizon):
        raise ValueError("First-hit/stopping fields disagree")
    if any(not math.isfinite(row[name]) or row[name] < 0 for name in ("a_cost", "b_cost")):
        raise ValueError("Invalid latent score")
    return dict(row, c_success=success, c_first_hit_step=hit, c_remaining_distance=distance,
                task_end_distance=distance, end_distance=row["full_horizon_goal_distance_m"], c_scope=scope)


def analyze_pool(occurrences, scope):
    rows = [c_outcome(row,scope) for row in occurrences]
    if len({(row["episode"],row["plan_call"]) for row in rows}) != 1:
        raise ValueError("A pool cannot mix decision states")
    if sorted(row["candidate_index"] for row in rows) != list(range(len(rows))):
        raise ValueError("Incomplete candidate occurrence indices")
    ranked = rank_candidates(deduplicate_routes(rows))
    lookup = {row["route_id"]:row for row in ranked}
    deployed = [row for row in rows if row["is_deployed_output"]]
    if len(deployed) != 1:
        raise ValueError("Expected one actual A final mean")
    choices = dict(a_deployed_mean=deployed[0], a_pool_min=min(rows,key=lambda row:row["a_cost"]),
                   b_pool_min=min(rows,key=lambda row:row["b_cost"]), c_pool_best=min(ranked,key=c_key))
    first = rows[0]
    base = dict(episode=first["episode"],plan_call=first["plan_call"],c_scope=scope,
                episode_steps_elapsed=first["episode_steps_elapsed"],episode_steps_left=first["episode_steps_left"],
                evaluation_horizon=first["prefix_steps"] if scope=="executed_prefix" else first["full_horizon_steps"])
    summary = dict(base,occurrences=len(rows),unique_routes=len(ranked),
                   successful_unique_routes=sum(row["c_success"] for row in ranked),
                   c_optimal_routes=sum(c_key(row)==c_key(choices["c_pool_best"]) for row in ranked),
                   **rank_correlations(ranked))
    selections = []
    for label,choice in choices.items():
        rank = lookup[choice["route_id"]]
        regret = regret_against_pool(choice,choices["b_pool_min"],choices["c_pool_best"])
        selections.append(dict(base,selection=label,candidate=choice["candidate"],candidate_index=choice["candidate_index"],
                               route_id=choice["route_id"],cem_round=choice["cem_round"],a_cost=choice["a_cost"],b_cost=choice["b_cost"],
                               c_success=choice["c_success"],c_first_hit_step=choice["c_first_hit_step"],
                               stopped_goal_distance_m=choice["c_remaining_distance"],
                               a_rank=rank["a_rank"],b_rank=rank["b_rank"],c_rank=rank["c_rank"],
                               b_regret=regret["b_regret"],c_optimal=regret["c_optimal"],
                               c_missed_success=regret["c_missed_success"],c_excess_steps=regret["c_excess_steps"],
                               c_excess_distance_m=regret["c_excess_distance"]))
        summary[label+"_success"] = choice["c_success"]
    for target in ("a_deployed_mean","a_pool_min"):
        b, a = c_key(choices["b_pool_min"]), c_key(choices[target])
        summary["b_vs_"+target] = "better" if b<a else "worse" if b>a else "tied"
    round_summaries, cumulative = [], []
    for _,batch in sorted(group_by(rows,"cem_round").items()):
        cumulative.extend(batch)
        round_summaries.append(dict(base,**summarize_round(batch,cumulative)))
    return ranked,summary,selections,round_summaries


def expected_pools(run):
    args = yaml.safe_load((run/"args_eval.yaml").read_text())
    pools = set()
    for ep in range(args["meta"]["eval_episodes"]):
        folder = run/"mw-reach-wall"/f"ep_{ep}"
        with (folder/"execution.csv").open(newline="") as handle:
            calls = {int(row["plan_call"]) for row in csv.DictReader(handle)}
        for call in calls:
            if not (folder/"branch_a"/f"plan_{call:03d}.pt").exists():
                raise ValueError("A decision bundle is missing")
            pools.add((ep,call))
    return pools


def aggregate_results(pools,selections,expected):
    result = dict(completed_pools=len(pools)//2,expected_pools=expected,
                  complete=len(pools)//2==expected,scopes={})
    for scope in ("full_horizon","executed_prefix"):
        batch = [row for row in pools if row["c_scope"]==scope]
        selected = [row for row in selections if row["c_scope"]==scope]
        summary = dict(pools=len(batch),pools_with_success=sum(row["successful_unique_routes"]>0 for row in batch),
                       mean_spearman={},selections={},comparisons={})
        for pair in ("ab","ac","bc"):
            values = [row["spearman_"+pair] for row in batch if row["spearman_"+pair] is not None]
            summary["mean_spearman"][pair] = float(np.mean(values)) if values else None
        for label in LABELS:
            choices = [row for row in selected if row["selection"]==label]
            summary["selections"][label] = dict(successes=sum(row["c_success"] for row in choices),
                c_optimal=sum(row["c_optimal"] for row in choices),missed_successes=sum(row["c_missed_success"] for row in choices))
        for target in ("a_deployed_mean","a_pool_min"):
            summary["comparisons"]["b_vs_"+target] = {value:sum(row["b_vs_"+target]==value for row in batch) for value in ("better","tied","worse")}
        result["scopes"][scope] = summary
    return result


def write_report(output,aggregate,pools,selections):
    n,expected = aggregate["completed_pools"],aggregate["expected_pools"]
    lines = ["# Reach-Wall decision-local A/B/C results", "",
             f"Coverage: **{n}/{expected} decision pools**. " + ("Complete." if aggregate["complete"] else "PARTIAL: results cover only completed B pools; unfinished pools are excluded."), "",
             "C ranks success first (native TCP distance <=0.05 m), then earliest first-hit action. "
             "Failures rank by terminal TCP distance. Successful routes arriving at the same step tie, regardless of endpoint distance. "
             "Steps and metres are never combined into a scalar.", "",
             "The failure distance is a ground-truth measurement and a proxy for progress; it does not account for obstacle detours or certify optimal paths. "
             "C selects the best observed candidate in this pool, not a global optimum.", ""]
    for scope,data in aggregate["scopes"].items():
        lines += ["## "+scope, "", f"Pools containing a successful candidate: {data['pools_with_success']}/{data['pools']}.", "",
                  "| Selection | Successful local routes | C-optimal choices | Missed available successes |",
                  "| --- | ---: | ---: | ---: |"]
        for name,values in data["selections"].items():
            lines.append(f"| {name} | {values['successes']}/{n} | {values['c_optimal']}/{n} | {values['missed_successes']} |")
        lines += ["", "Mean within-pool Spearman correlations: " + ", ".join(f"{pair.upper()}={value:.4f}" if value is not None else f"{pair.upper()}=undefined" for pair,value in data["mean_spearman"].items())+".", ""]
        for target,counts in data["comparisons"].items():
            lines.append(f"{target}: B better in {counts['better']} pools, tied in {counts['tied']}, worse in {counts['worse']}.")
        lines += ["", "| Planning call | Pools | Pool contains success | A deployed | Minimum A | Minimum B |", "| ---: | ---: | ---: | ---: | ---: | ---: |"]
        for call,group in sorted(group_by([row for row in pools if row["c_scope"]==scope],"plan_call").items()):
            lines.append(f"| {call} | {len(group)} | {sum(row['successful_unique_routes']>0 for row in group)} | " + " | ".join(str(sum(row[label+'_success'] for row in group)) for label in LABELS[:3])+" |")
        lines.append("")
    lines += ["## Interpretation and limits", "",
              "- Counts describe candidate selections at A's saved starting states, not episode success rates of independently deployed B/C controllers.",
              "- The primary full_horizon view matches A/B's latent prediction horizon. The executed_prefix view diagnoses the shorter prefix actually returned before replanning; A/B selection costs remain their full-horizon costs.",
              "- Exact action duplicates are collapsed within each episode/plan-call pool for rankings. Occurrence metadata is retained for round summaries; tied ranks are averaged.",
              "- Planning calls within an episode are related. No candidate-level significance tests or independence claims are made.",
              "- B/C disagreement includes latent geometry, objective choice, and endpoint versus first-hit timing. It does not alone establish encoder failure.",
              "- Conditional regret is split into missed success, excess first-hit steps for successful choices, and excess terminal distance only when the pool contains no success.",
              "- Partial results may be biased by which early episodes finished. Read coverage before interpreting aggregates.", ""]
    (output/"results.md").write_text("\n".join(lines),encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path,default=DEFAULT_RUN)
    parser.add_argument("--output-dir",type=Path)
    parser.add_argument("--allow-partial",action="store_true")
    parser.add_argument("--write-ranked",action="store_true",help="Export all unique-route ranks; large CSV")
    args = parser.parse_args(argv)
    run = args.run_dir.resolve()
    output = args.output_dir or run/("branch_c_partial" if args.allow_partial else "branch_c")
    expected = expected_pools(run)
    connection = sqlite3.connect("file:"+(run/"branch_b/scores.sqlite3").as_posix()+"?mode=ro",uri=True)
    connection.execute("BEGIN")  # A consistent snapshot if B is still appending.
    counts = connection.execute("SELECT p.episode,p.plan_call,p.total,count(s.candidate_index) FROM pools p LEFT JOIN scores s ON p.episode=s.episode AND p.plan_call=s.plan_call GROUP BY p.episode,p.plan_call ORDER BY p.episode,p.plan_call").fetchall()
    complete = [(ep,call,total) for ep,call,total,count in counts if total==count]
    if any((ep,call) not in expected for ep,call,_,_ in counts):
        connection.close()
        raise ValueError("B contains unexpected decision-state identifiers")
    if not args.allow_partial and {(ep,call) for ep,call,_ in complete} != expected:
        connection.close()
        raise ValueError(f"B is incomplete: {len(complete)}/{len(expected)} pools. Complete B first, or explicitly use --allow-partial.")
    if not complete:
        connection.close()
        raise ValueError("No complete B pools")
    producer_protocol=json.loads(connection.execute("SELECT value FROM metadata WHERE key='protocol'").fetchone()[0])
    output.mkdir(parents=True,exist_ok=True)
    pools,selections,rounds = [],[],[]
    ranked_handle = (output/"ranked_candidates.csv.tmp").open("w",newline="",encoding="utf-8") if args.write_ranked else None
    writer = None
    try:
        for index,(ep,call,total) in enumerate(complete):
            raw = [json.loads(payload) for (payload,) in connection.execute("SELECT payload FROM scores WHERE episode=? AND plan_call=? ORDER BY candidate_index",(ep,call))]
            if len(raw)!=total:
                raise ValueError("Pool changed during analysis")
            for scope in ("full_horizon","executed_prefix"):
                ranked,summary,chosen,per_round = analyze_pool(raw,scope)
                pools.append(summary);selections.extend(chosen);rounds.extend(per_round)
                if ranked_handle is not None:
                    for row in ranked:
                        compact = {key:row[key] for key in ("episode","plan_call","c_scope","candidate","route_id","occurrence_count","a_cost","b_cost","a_rank","b_rank","c_rank","c_success","c_first_hit_step","c_remaining_distance")}
                        if writer is None:
                            writer=csv.DictWriter(ranked_handle,fieldnames=list(compact));writer.writeheader()
                        writer.writerow(compact)
            if index%20==0 or index+1==len(complete):
                print(f"C analysis: {index+1}/{len(complete)} complete pools",flush=True)
    finally:
        connection.close()
        if ranked_handle is not None:
            ranked_handle.close()
    if args.write_ranked:
        (output/"ranked_candidates.csv.tmp").replace(output/"ranked_candidates.csv")
    aggregate=aggregate_results(pools,selections,len(expected))
    protocol=dict(c_ordering="success first; earliest hit for successes; terminal TCP distance for failures",
                  native_success_threshold_m=.05,failure_distance_is_obstacle_aware=False,
                  primary_scope="full_horizon",secondary_scope="executed_prefix",
                  source_database=str(run/"branch_b/scores.sqlite3"),producer_protocol=producer_protocol,
                  analyzed_pools=[dict(episode=ep,plan_call=call,occurrences=total) for ep,call,total in complete],
                  expected_pools=len(expected),complete=aggregate["complete"],
                  deduplication="within episode/plan_call; first occurrence canonical; average tied ranks")
    (output/"protocol.json").write_text(json.dumps(protocol,indent=2),encoding="utf-8")
    for filename,rows in (("pool_summary.csv",pools),("selections.csv",selections),("round_summary.csv",rounds)):
        write_csv(output/filename,rows)
    write_csv(output/"planning_regret.csv",[row for row in selections if row["selection"]!="c_pool_best"])
    (output/"summary.json").write_text(json.dumps(aggregate,indent=2),encoding="utf-8")
    write_report(output,aggregate,pools,selections)
    print(json.dumps(aggregate,indent=2),flush=True)
    print("Results:",output/"results.md",flush=True)


if __name__=="__main__":
    main()

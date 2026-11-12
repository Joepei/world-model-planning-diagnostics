"""Compare A/B proposal quality at their paired first planning decisions.

Common native outcomes measure search coverage, proposal distributions, and
selection losses separately. Related candidates are not independent trials;
paired uncertainty resamples the episode, not candidate rows.
"""

import argparse
import csv
import json
from pathlib import Path
import sqlite3

import numpy as np
import torch

from .replay_reach_wall_prefix_candidates import DEFAULT_RUNS,route_id,sha,write_json


SCOPES=("full","prefix")
BOOTSTRAP_SEED=20261108


def csv_file(path,rows):
    with path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def a_record(row):
    return dict(method="A",episode=row["episode"],candidate=row["candidate"],candidate_index=row["candidate_index"],
        route_id=row["route_id"],cem_round=row["cem_round"],origin=row["origin"],is_elite=row["is_elite"],
        is_deployed_output=row["is_deployed_output"],own_cost=float(row["a_cost"]),
        full_success=bool(row["task_success"]),full_first_hit=row["task_first_hit_step"],
        full_distance_cm=row["task_end_distance_m"]*100,full_endpoint_distance_cm=row["full_horizon_goal_distance_m"]*100,
        full_endpoint_success=bool(row["full_horizon_success"]),prefix_success=bool(row["prefix_success"]),
        prefix_first_hit=row["task_first_hit_step"] if row["prefix_success"] else None,
        prefix_distance_cm=row["prefix_task_end_distance_m"]*100)


def b_record(episode,index,candidate,outcome):
    hit=outcome["full_first_hit"]
    stopped=outcome["distance_trace_m"][hit-1] if hit else outcome["full_endpoint_distance_m"]
    return dict(method="B",episode=episode,candidate=candidate["name"],candidate_index=index,
        route_id=route_id(candidate["actions_normalized"].numpy()),cem_round=candidate["cem_round"],
        origin=candidate["origin"],is_elite=candidate["is_elite"],is_deployed_output=candidate["is_deployed_output"],
        own_cost=float(candidate["b_cost"]),full_success=bool(outcome["full_success"]),full_first_hit=hit,
        full_distance_cm=stopped*100,full_endpoint_distance_cm=outcome["full_endpoint_distance_m"]*100,
        full_endpoint_success=bool(outcome["full_endpoint_success"]),prefix_success=bool(outcome["prefix_success"]),
        prefix_first_hit=outcome["prefix_first_hit"],prefix_distance_cm=outcome["prefix_stopped_distance_m"]*100)


def task_key(row,scope):
    return (0,row[f"{scope}_first_hit"]) if row[f"{scope}_success"] else (1,row[f"{scope}_distance_cm"])


def compare(a,b,scope):
    """B wins when smaller; failed distances within 1e-6 cm are tied."""
    ka,kb=task_key(a,scope),task_key(b,scope)
    if ka[0]==kb[0]==1 and abs(ka[1]-kb[1])<=1e-6:return "tie"
    return "B" if kb<ka else "A" if ka<kb else "tie"


def regret(row,best,scope):
    rs,bs=row[f"{scope}_success"],best[f"{scope}_success"]
    return dict(missed_success=bool(bs and not rs),
        excess_first_hit_actions=row[f"{scope}_first_hit"]-best[f"{scope}_first_hit"] if rs and bs else None,
        excess_failure_distance_cm=row[f"{scope}_distance_cm"]-best[f"{scope}_distance_cm"] if not bs else None)


def selected_row(row,label,scope,best):
    return dict(method=row["method"],episode=row["episode"],scope=scope,selection=label,candidate=row["candidate"],
        candidate_index=row["candidate_index"],route_id=row["route_id"],cem_round=row["cem_round"],
        own_cost=row["own_cost"],success=row[f"{scope}_success"],first_hit=row[f"{scope}_first_hit"],
        stopped_distance_cm=row[f"{scope}_distance_cm"],**regret(row,best,scope))


def describe_pool(records):
    samples=[r for r in records if r["origin"]!="updated_mean"]
    unique={r["route_id"]:r for r in records}
    if len(records)!=4515 or len(samples)!=4500 or len(unique)!=4501:
        raise ValueError("Unexpected or incomplete proposal budget")
    chosen=[r for r in records if r["is_deployed_output"]]
    if len(chosen)!=1 or chosen[0]["candidate"]!="mean_round_15":
        raise ValueError("Actual deployed mean is missing")
    summary=dict(method=records[0]["method"],episode=records[0]["episode"],captured_occurrences=len(records),
        sample_occurrences=len(samples),unique_routes=len(unique),
        full_endpoint_successful_samples=sum(r["full_endpoint_success"] for r in samples),
        terminal_best_distance_cm=min(r["full_endpoint_distance_cm"] for r in unique.values()))
    choices={"returned_mean":chosen[0],"minimum_own_cost_sample":min(samples,key=lambda r:r["own_cost"]),
             "minimum_own_cost_captured":min(records,key=lambda r:r["own_cost"])}
    selected=[]
    for scope in SCOPES:
        good=sum(r[f"{scope}_success"] for r in samples)
        post=[r for r in samples if r["cem_round"]>1]
        best=min(unique.values(),key=lambda r:task_key(r,scope))
        summary.update({
            f"{scope}_successful_samples":good,f"{scope}_sample_success_pct":100*good/len(samples),
            f"{scope}_successful_post_round1_samples":sum(r[f"{scope}_success"] for r in post),
            f"{scope}_post_round1_success_pct":100*sum(r[f"{scope}_success"] for r in post)/len(post),
            f"{scope}_successful_unique_routes":sum(r[f"{scope}_success"] for r in unique.values()),
            f"{scope}_pool_contains_success":best[f"{scope}_success"],
            f"{scope}_best_first_hit":best[f"{scope}_first_hit"],f"{scope}_best_distance_cm":best[f"{scope}_distance_cm"],
            f"{scope}_mean_sample_distance_cm":float(np.mean([r[f"{scope}_distance_cm"] for r in samples])),
            f"{scope}_median_sample_distance_cm":float(np.median([r[f"{scope}_distance_cm"] for r in samples])),
            f"{scope}_returned_mean_success":chosen[0][f"{scope}_success"],
            f"{scope}_returned_mean_first_hit":chosen[0][f"{scope}_first_hit"],
            f"{scope}_returned_mean_distance_cm":chosen[0][f"{scope}_distance_cm"],
            f"{scope}_returned_mean_missed_success":bool(best[f"{scope}_success"] and not chosen[0][f"{scope}_success"])})
        for label,row in dict(choices,native_pool_best=best).items():
            selected.append(selected_row(row,label,scope,best))
    rounds=[]
    cumulative=[]
    for number in range(1,16):
        sampled=[r for r in samples if r["cem_round"]==number]
        all_round=[r for r in records if r["cem_round"]==number]
        if len(sampled)!=300 or len(all_round)!=301:raise ValueError("Incomplete CEM round")
        cumulative.extend(all_round)
        result=dict(method=records[0]["method"],episode=records[0]["episode"],cem_round=number,samples=300)
        for scope in SCOPES:
            best=min(cumulative,key=lambda r:task_key(r,scope))
            result.update({f"{scope}_successful_samples":sum(r[f"{scope}_success"] for r in sampled),
                f"{scope}_sample_success_pct":100*sum(r[f"{scope}_success"] for r in sampled)/300,
                f"{scope}_mean_sample_distance_cm":float(np.mean([r[f"{scope}_distance_cm"] for r in sampled])),
                f"{scope}_cumulative_pool_has_success":best[f"{scope}_success"],
                f"{scope}_cumulative_best_first_hit":best[f"{scope}_first_hit"],
                f"{scope}_cumulative_best_distance_cm":best[f"{scope}_distance_cm"]})
        rounds.append(result)
    return summary,selected,rounds


def paired_bootstrap(values):
    values=np.asarray(values,dtype=float)
    rng=np.random.default_rng(BOOTSTRAP_SEED)
    estimates=values[rng.integers(0,len(values),size=(20000,len(values)))].mean(axis=1)
    return dict(mean=float(values.mean()),median=float(np.median(values)),
        paired_episode_bootstrap_95_percentile_ci=np.quantile(estimates,[.025,.975]).tolist())


def aggregate_rounds(rows):
    output=[]
    for method in ("A","B"):
        for number in range(1,16):
            group=[r for r in rows if r["method"]==method and r["cem_round"]==number]
            row=dict(method=method,cem_round=number,episodes=len(group),samples=300*len(group))
            for scope in SCOPES:
                count=sum(r[f"{scope}_successful_samples"] for r in group)
                row.update({f"{scope}_successful_samples":count,f"{scope}_sample_success_pct":100*count/row["samples"],
                    f"{scope}_cumulative_covered_episodes":sum(r[f"{scope}_cumulative_pool_has_success"] for r in group),
                    f"{scope}_mean_sample_distance_cm":float(np.mean([r[f"{scope}_mean_sample_distance_cm"] for r in group])),
                    f"{scope}_median_cumulative_best_distance_cm":float(np.median([r[f"{scope}_cumulative_best_distance_cm"] for r in group]))})
            output.append(row)
    return output


def plot_rounds(rows,output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with plt.rc_context({"font.size":10,"pdf.fonttype":42}):
        fig,axes=plt.subplots(1,3,figsize=(15,5.2))
        for method,color in (("A","#0072B2"),("B","#D55E00")):
            group=[r for r in rows if r["method"]==method]
            x=[r["cem_round"] for r in group]
            axes[0].plot(x,[r["full_sample_success_pct"] for r in group],"o-",color=color,ms=4,label=method)
            axes[1].plot(x,[r["full_cumulative_covered_episodes"] for r in group],"o-",color=color,ms=4,label=method)
            axes[2].plot(x,[r["prefix_mean_sample_distance_cm"] for r in group],"o-",color=color,ms=4,label=method)
        axes[0].set(title="Full 30-action sample success",ylabel="Successful sample sequences (%)")
        axes[1].set(title="Accumulated full-plan coverage",ylabel="Initial states with a successful candidate",ylim=(-1,49))
        axes[2].set(title="Quality of the 15-action prefix",ylabel="Mean native distance after prefix (cm)")
        for ax in axes:
            ax.set(xlabel="CEM round",xticks=[1,3,5,7,9,11,13,15],xlim=(.7,15.3))
            ax.spines[["top","right"]].set_visible(False)
            ax.grid(color=".9",lw=.7);ax.legend(frameon=False)
        fig.suptitle("A versus B: search from the same 48 initial states",fontsize=15,y=.98)
        fig.text(.5,.012,"A: learned-dynamics-guided search. B: oracle-dynamics-guided search. Same horizon and proposal budget; first populations are identical.\n"
            "Native task outcomes use first-hit success (TCP distance <=5 cm). Full-plan coverage includes sampled sequences and updated means.",
            ha="center",fontsize=8.5,color=".3")
        fig.subplots_adjust(left=.055,right=.99,top=.81,bottom=.25,wspace=.30)
        for suffix in ("png","pdf"):
            fig.savefig(output/f"initial_search_comparison.{suffix}",dpi=190,bbox_inches="tight")
        plt.close(fig)


def aggregate_methods(pools,selections):
    result={}
    for method in ("A","B"):
        group=[r for r in pools if r["method"]==method]
        result[method]=dict(episodes=len(group),sample_occurrences=sum(r["sample_occurrences"] for r in group),
            unique_state_route_pairs=sum(r["unique_routes"] for r in group),
            full_endpoint_successful_samples=sum(r["full_endpoint_successful_samples"] for r in group),
            median_terminal_best_distance_cm=float(np.median([r["terminal_best_distance_cm"] for r in group])))
        for scope in SCOPES:
            count=sum(r[f"{scope}_successful_samples"] for r in group)
            native=[r for r in selections if r["method"]==method and r["scope"]==scope]
            result[method][scope]=dict(successful_samples=count,sample_success_pct=100*count/result[method]["sample_occurrences"],
                pool_covered_episodes=sum(r[f"{scope}_pool_contains_success"] for r in group),
                returned_mean_successes=sum(r[f"{scope}_returned_mean_success"] for r in group),
                returned_mean_missed_successes=sum(r[f"{scope}_returned_mean_missed_success"] for r in group),
                minimum_own_cost_sample_successes=sum(r["success"] for r in native if r["selection"]=="minimum_own_cost_sample"),
                mean_sample_stopped_distance_cm=float(np.mean([r[f"{scope}_mean_sample_distance_cm"] for r in group])),
                median_best_stopped_distance_cm=float(np.median([r[f"{scope}_best_distance_cm"] for r in group])),
                mean_returned_mean_stopped_distance_cm=float(np.mean([r[f"{scope}_returned_mean_distance_cm"] for r in group])),
                post_round1_sample_success_pct=float(np.mean([r[f"{scope}_post_round1_success_pct"] for r in group])))
    return result


def make_report(output,summary):
    n=summary["completed_episodes"]
    a,b=summary["methods"]["A"],summary["methods"]["B"]
    lines=["# A/B search competency at paired initial states", "",
        f"Coverage: {n}/48 paired initial decisions. "+("Complete." if summary["complete"] else "PARTIAL; early completed episodes may not represent the full set."), "",
        "A uses learned predictive dynamics to guide CEM; B independently adapts CEM using simulator-derived endpoint embeddings. Both start at the same physical state and goal with horizon 30, prefix limit 15, 15 rounds, 300 evaluated samples per round, ten elites, native bounded actions, and the same first population. Native simulator measurements evaluate both pools.", "",
        "## Full 30-action planning horizon", "",
        "| Measure | A | B |", "|---|---:|---:|"]
    for label,key in (("Successful evaluated sample sequences","successful_samples"),("Sample success rate (%)","sample_success_pct"),
            ("Initial pools containing a successful route","pool_covered_episodes"),("Returned final means that would succeed if completed","returned_mean_successes"),
            ("Lowest-own-cost samples that would succeed","minimum_own_cost_sample_successes"),
            ("Returned means missing an available successful route","returned_mean_missed_successes")):
        x,y=a["full"][key],b["full"][key]
        lines.append(f"| {label} | {x:.4f} | {y:.4f} |" if isinstance(x,float) else f"| {label} | {x} | {y} |")
    full=summary["paired"]["full"]
    interval=full["sample_success_pct_difference_B_minus_A"]["paired_episode_bootstrap_95_percentile_ci"]
    lines += ["",f"Samples whose raw action-30 endpoint is inside the native goal region: A {a['full_endpoint_successful_samples']}, B {b['full_endpoint_successful_samples']}. This is a terminal-only control alongside first-hit task success.", "",
        f"Best native route in each pool: B better in {full['pool_best_wins']['B']} pairs, A better in {full['pool_best_wins']['A']}, tied in {full['pool_best_wins']['tie']}. Comparisons prioritize success, then earlier first hit; failed routes use terminal TCP distance.", "",
        f"Mean within-episode sample-success-rate difference (B minus A): {full['sample_success_pct_difference_B_minus_A']['mean']:.4f} percentage points. Exploratory 95% paired-episode bootstrap interval: [{interval[0]:.4f}, {interval[1]:.4f}] percentage points.", "",
        "## Actual 15-action execution prefix", "", "| Measure | A | B |", "|---|---:|---:|"]
    for label,key in (("Successful evaluated sample prefixes","successful_samples"),("Pools containing a successful prefix","pool_covered_episodes"),
        ("Returned means whose prefixes succeed","returned_mean_successes"),("Mean sampled-prefix distance (cm)","mean_sample_stopped_distance_cm"),
        ("Mean returned-prefix distance (cm)","mean_returned_mean_stopped_distance_cm"),("Median best-prefix distance (cm)","median_best_stopped_distance_cm")):
        x,y=a["prefix"][key],b["prefix"][key]
        lines.append(f"| {label} | {x:.3f} | {y:.3f} |" if isinstance(x,float) else f"| {label} | {x} | {y} |")
    p=summary["paired"]["prefix"]
    lines += ["",f"Best native prefix: B better in {p['pool_best_wins']['B']} pairs, A better in {p['pool_best_wins']['A']}, tied in {p['pool_best_wins']['tie']}.", "",
        "## Controls and interpretation", "",
        f"- All {n} physics snapshots and goal encodings match. The first 300 normalized and raw sample actions match exactly in each episode. Native results match on {summary['pairing_controls']['shared_native_routes_checked']} shared routes, with maximum distance difference {summary['pairing_controls']['max_native_error_m']:.3g} m.",
        "- Each method contributes 4,500 sample occurrences per episode (including the inserted distribution mean) and 15 separately recorded updated means. Success fractions use the 4,500 samples; coverage/best-option measures include all captured candidates. Exact duplicate routes are collapsed for pool coverage and union references.",
        "- Native success means TCP distance <=5 cm at any action within the relevant horizon. Successful routes stop logically at first hit. Raw action-30 terminal success/distance is also retained as a control.",
        "- Failed-route TCP distance is a proximity proxy. It does not measure an obstacle-aware remaining route or certify that a closer failure state is easier to solve.",
        "- Search quality is measured by pool coverage, success fractions, and native pool-best quality. Returning the final mean is evaluated separately. A larger pool potential does not establish a better deployed controller.",
        "- Full 30-action outcomes are counterfactual after the first 15 actions. This study is at initial states only; it does not compare later states, change replanning, or rerun full episodes.",
        "- The combined observed A/B pool is a finite reference, not a global oracle over all admissible action sequences. Search/selection regret separates missed success, excess first-hit steps, and excess distance only when the reference has no success. No mixed-unit scalar is formed.",
        "- Equal proposal budgets do not imply equal runtime or simulator compute. This study does not benchmark computational efficiency.",
        "- Bootstrap intervals resample paired episode IDs (20,000 replicates, fixed seed 20261108), not the dependent candidate occurrences. They describe this 48-start pilot and do not replace repeated search seeds or environment samples.", ""]
    (output/"results.md").write_text("\n".join(lines),encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir",type=Path,default=DEFAULT_RUNS)
    parser.add_argument("--output-dir",type=Path)
    parser.add_argument("--allow-partial",action="store_true")
    parser.add_argument("--write-candidates",action="store_true",help="Export the large standardized candidate CSV")
    args=parser.parse_args()
    runs=args.runs_dir.resolve()
    output=(args.output_dir or runs.parent/"analysis/initial_search_ab").resolve()
    source_a=runs/"bounded_a_i15_n300_ep48/branch_b/scores.sqlite3"
    adb=sqlite3.connect("file:"+source_a.as_posix()+"?mode=ro",uri=True)
    bdb=sqlite3.connect("file:"+(output/"scores.sqlite3").as_posix()+"?mode=ro",uri=True)
    adb.execute("BEGIN");bdb.execute("BEGIN")
    source_protocol=json.loads(bdb.execute("SELECT value FROM metadata WHERE key='protocol'").fetchone()[0])
    completed=bdb.execute("SELECT episode,total,control FROM pools WHERE complete=1 ORDER BY episode").fetchall()
    if not args.allow_partial and len(completed)!=48:raise ValueError(f"Initial B scoring incomplete: {len(completed)}/48")
    if not completed:raise ValueError("No completed initial pools")
    pool_rows,selections,rounds,paired,search_regrets,controls=[],[],[],[],[],[]
    candidate_handle=(output/"candidate_quality.csv").open("w",newline="",encoding="utf-8") if args.write_candidates else None
    candidate_writer=None
    try:
        for ep,total,control_json in completed:
            control=json.loads(control_json);controls.append(control)
            raw_a=[json.loads(row[0]) for row in adb.execute("SELECT payload FROM scores WHERE episode=? AND plan_call=0 ORDER BY candidate_index",(ep,))]
            original=adb.execute("SELECT source_sha256,total FROM pools WHERE episode=? AND plan_call=0",(ep,)).fetchone()
            if original!=(control["a_sha256"],4515):raise ValueError("A source/replay provenance mismatch")
            records_a=[a_record(r) for r in raw_a]
            b_path=runs/f"closed_loop_b/ep_{ep}/plan_000.pt"
            if sha(b_path)!=control["b_sha256"]:raise ValueError("B source pool changed")
            saved_b=torch.load(b_path,map_location="cpu",weights_only=False)
            outcomes={key:json.loads(payload) for key,payload in bdb.execute("SELECT route_id,payload FROM scores WHERE episode=?",(ep,))}
            if len(outcomes)!=total:raise ValueError("B native scoring is incomplete")
            records_b=[b_record(ep,i,c,outcomes[route_id(c["actions_simulator"].numpy())]) for i,c in enumerate(saved_b["candidates"])]
            for records in (records_a,records_b):
                pool,selected,by_round=describe_pool(records)
                pool["initial_tcp_goal_distance_cm"]=control["initial_tcp_goal_distance_m"]*100
                pool_rows.append(pool);selections.extend(selected);rounds.extend(by_round)
                if candidate_handle:
                    if candidate_writer is None:
                        candidate_writer=csv.DictWriter(candidate_handle,fieldnames=list(records[0]));candidate_writer.writeheader()
                    candidate_writer.writerows(records)
            lookup_a={r["route_id"]:r for r in records_a}
            lookup_b={r["route_id"]:r for r in records_b}
            overlap=set(lookup_a)&set(lookup_b)
            if len(overlap)!=control["counts"]["shared_native_routes_checked"]:raise ValueError("Shared-route control count differs")
            for key in overlap:
                x,y=lookup_a[key],lookup_b[key]
                for scope in SCOPES:
                    if x[f"{scope}_success"]!=y[f"{scope}_success"] or x[f"{scope}_first_hit"]!=y[f"{scope}_first_hit"]:
                        raise ValueError("Shared-route native outcomes differ")
                    if abs(x[f"{scope}_distance_cm"]-y[f"{scope}_distance_cm"])>1e-6:raise ValueError("Shared-route native distances differ")
            summaries={r["method"]:r for r in pool_rows if r["episode"]==ep}
            union=dict(lookup_a);union.update(lookup_b)
            for scope in SCOPES:
                a_best=min(lookup_a.values(),key=lambda r:task_key(r,scope))
                b_best=min(lookup_b.values(),key=lambda r:task_key(r,scope))
                union_best=min(union.values(),key=lambda r:task_key(r,scope))
                a_chosen=next(r for r in records_a if r["is_deployed_output"])
                b_chosen=next(r for r in records_b if r["is_deployed_output"])
                paired.append(dict(episode=ep,scope=scope,shared_unique_routes=len(overlap),union_unique_routes=len(union),
                    initial_tcp_goal_distance_cm=control["initial_tcp_goal_distance_m"]*100,
                    a_pool_contains_success=a_best[f"{scope}_success"],b_pool_contains_success=b_best[f"{scope}_success"],
                    a_sample_success_pct=summaries["A"][f"{scope}_sample_success_pct"],b_sample_success_pct=summaries["B"][f"{scope}_sample_success_pct"],
                    sample_success_pct_delta_B_minus_A=summaries["B"][f"{scope}_sample_success_pct"]-summaries["A"][f"{scope}_sample_success_pct"],
                    mean_sample_distance_delta_B_minus_A_cm=summaries["B"][f"{scope}_mean_sample_distance_cm"]-summaries["A"][f"{scope}_mean_sample_distance_cm"],
                    a_best_first_hit=a_best[f"{scope}_first_hit"],b_best_first_hit=b_best[f"{scope}_first_hit"],
                    a_best_distance_cm=a_best[f"{scope}_distance_cm"],b_best_distance_cm=b_best[f"{scope}_distance_cm"],
                    pool_best_winner=compare(a_best,b_best,scope),returned_mean_winner=compare(a_chosen,b_chosen,scope),
                    a_returned_success=a_chosen[f"{scope}_success"],b_returned_success=b_chosen[f"{scope}_success"]))
                for method,best in (("A",a_best),("B",b_best)):
                    search_regrets.append(dict(method=method,episode=ep,scope=scope,reference="best of combined observed A/B pool",
                        pool_best_candidate=best["candidate"],reference_method=union_best["method"],
                        reference_candidate=union_best["candidate"],**regret(best,union_best,scope)))
    finally:
        adb.close();bdb.close()
        if candidate_handle:candidate_handle.close()
    for filename,rows in (("pool_summary.csv",pool_rows),("selections.csv",selections),("round_summary.csv",rounds),
                           ("paired_comparison.csv",paired),("conditional_search_regret.csv",search_regrets)):
        csv_file(output/filename,rows)
    csv_file(output/"selection_regret.csv",[r for r in selections if r["selection"]=="returned_mean"])
    round_totals=aggregate_rounds(rounds)
    csv_file(output/"round_aggregate.csv",round_totals)
    comparisons={}
    for scope in SCOPES:
        group=[r for r in paired if r["scope"]==scope]
        comparisons[scope]=dict(pool_best_wins={label:sum(r["pool_best_winner"]==label for r in group) for label in ("A","B","tie")},
            returned_mean_wins={label:sum(r["returned_mean_winner"]==label for r in group) for label in ("A","B","tie")},
            sample_success_pct_difference_B_minus_A=paired_bootstrap([r["sample_success_pct_delta_B_minus_A"] for r in group]),
            mean_sample_distance_difference_B_minus_A_cm=paired_bootstrap([r["mean_sample_distance_delta_B_minus_A_cm"] for r in group]))
    summary=dict(completed_episodes=len(completed),expected_episodes=48,complete=len(completed)==48,
        methods=aggregate_methods(pool_rows,selections),paired=comparisons,
        pairing_controls=dict(paired_initial_physics=True,paired_goal_encoding=True,matched_first_population_per_episode=300,
            shared_native_routes_checked=sum(r["counts"]["shared_native_routes_checked"] for r in controls),
            max_native_error_m=max(r["shared_native_max_error_m"] for r in controls)))
    write_json(output/"summary.json",summary)
    write_json(output/"analysis_protocol.json",dict(source_replay_protocol=source_protocol,complete=len(completed)==48,
        analyzed_episodes=[r[0] for r in completed],bootstrap_seed=BOOTSTRAP_SEED,bootstrap_replicates=20000,
        bootstrap_unit="paired episode ID",source_a_replay_database=str(source_a),candidate_export_requested=args.write_candidates,
        comparison="Independent first-call candidate generation; common native task scoring; no later-state comparisons"))
    make_report(output,summary)
    if len(completed)==48:plot_rounds(round_totals,output)
    print(json.dumps(summary,indent=2),flush=True)
    print(f"Results: {output/'results.md'}",flush=True)


if __name__=="__main__":main()

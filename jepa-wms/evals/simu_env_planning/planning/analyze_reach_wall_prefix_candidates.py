"""Describe native prefix coverage and B's original endpoint-cost selection.

Read cached native replays, unchanged B candidate costs, and recorded elite
membership. This does not rerun or alter any controller or scoring objective.
"""

import argparse
import csv
import json
from pathlib import Path
import sqlite3

import numpy as np
import torch

from .replay_reach_wall_prefix_candidates import DEFAULT_RUNS,route_id,sha,write_json


def csv_file(path,rows,fields=None):
    fields=fields or list(rows[0])
    with path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def ranks(values):
    values=np.asarray(values)
    order=np.argsort(values,kind="stable")
    result=np.empty(len(values),dtype=float)
    start=0
    while start<len(order):
        stop=start+1
        while stop<len(order) and values[order[stop]]==values[order[start]]:stop+=1
        result[order[start:stop]]=(start+1+stop)/2
        start=stop
    return result


def analyze_pool(runs,ep,call,scores,control):
    path=runs/f"closed_loop_b/ep_{ep}/plan_{call:03d}.pt"
    if sha(path)!=control["source_sha256"]:
        raise ValueError("Source B candidate bundle changed")
    saved=torch.load(path,map_location="cpu",weights_only=False)
    candidates=saved["candidates"]
    cost_rank=ranks([c["b_cost"] for c in candidates])
    round_rank,cutoffs={},{}
    for round_number in range(1,16):
        indices=[i for i,c in enumerate(candidates) if c["cem_round"]==round_number and c["origin"]!="updated_mean"]
        ranked=ranks([candidates[i]["b_cost"] for i in indices])
        round_rank.update(zip(indices,map(float,ranked)))
        cutoffs[round_number]=max(candidates[i]["b_cost"] for i in indices if candidates[i]["is_elite"])
    chosen=next(c for c in candidates if c["is_deployed_output"])
    chosen_outcome=scores[route_id(chosen["actions_simulator"].numpy())]
    if chosen_outcome["prefix_success"]:
        raise ValueError("This diagnostic expects B's recorded failed prefixes")
    expanded=[]
    for index,candidate in enumerate(candidates):
        raw=candidate["actions_simulator"].numpy()
        key=route_id(raw)
        outcome=scores[key]
        if outcome["prefix_success"] and not outcome["full_outcome_available"]:
            raise ValueError("Missing full native continuation for a successful prefix")
        why=("intermediate updated mean; only final updated mean is deployed" if candidate["origin"]=="updated_mean" else
             "elite member; CEM deploys the elite action mean rather than a member" if candidate["is_elite"] else
             "outside its round's elites under the original full-horizon latent cost")
        if candidate["is_deployed_output"]:why="actual returned final mean"
        cutoff=cutoffs[candidate["cem_round"]] if candidate["origin"]!="updated_mean" else None
        if candidate["origin"]!="updated_mean" and not candidate["is_elite"] and candidate["b_cost"]==cutoff:
            why="tied at round elite cutoff; excluded by the original top-k tie choice"
        expanded.append(dict(episode=ep,plan_call=call,start_action=control["start_action"],candidate_index=index,
            candidate=candidate["name"],cem_round=candidate["cem_round"],origin=candidate["origin"],
            is_elite=candidate["is_elite"],is_deployed_output=candidate["is_deployed_output"],
            full_route_id=key,prefix_route_id=route_id(raw[:15]),b_cost=float(candidate["b_cost"]),
            b_cost_rank_among_occurrences=float(cost_rank[index]),prefix_success=outcome["prefix_success"],
            b_cost_rank_in_round=round_rank.get(index),round_elite_cutoff_b_cost=cutoff,
            cost_minus_round_elite_cutoff=candidate["b_cost"]-cutoff if cutoff is not None else None,
            prefix_first_hit=outcome["prefix_first_hit"],prefix_stopped_distance_cm=outcome["prefix_stopped_distance_m"]*100,
            prefix_endpoint_distance_cm=outcome["prefix_endpoint_distance_m"]*100,
            full_outcome_available=outcome["full_outcome_available"],full_success=outcome["full_success"],
            full_endpoint_success=outcome["full_endpoint_success"],
            full_endpoint_distance_cm=outcome["full_endpoint_distance_m"]*100 if outcome["full_outcome_available"] else None,
            cost_minus_returned_mean=float(candidate["b_cost"]-chosen["b_cost"]),
            original_return_rule_explanation=why))
    samples=[r for r in expanded if r["origin"]!="updated_mean"]
    gaussian=[r for r in samples if r["origin"]=="gaussian_sample"]
    good=[r for r in expanded if r["prefix_success"]]
    good_samples=[r for r in samples if r["prefix_success"]]
    last=[r for r in samples if r["cem_round"]==15]
    last_elites=[r for r in last if r["is_elite"]]
    prefix_lookup={r["prefix_route_id"]:r["prefix_success"] for r in expanded}
    lowest=min(expanded,key=lambda r:r["b_cost"])
    lowest_sample=min(samples,key=lambda r:r["b_cost"])
    best_good=min(good,key=lambda r:r["b_cost"]) if good else None
    terminal_good=[r for r in good if r["full_endpoint_success"]]
    if len(samples)!=4500 or len(gaussian)!=4485 or len(last_elites)!=10:
        raise ValueError("Unexpected search-budget or final-elite capture")
    rounds=[]
    for round_number in range(1,16):
        batch=[r for r in samples if r["cem_round"]==round_number]
        elites=[r for r in batch if r["is_elite"]]
        mean=next(r for r in expanded if r["origin"]=="updated_mean" and r["cem_round"]==round_number)
        if len(batch)!=300 or len(elites)!=10:
            raise ValueError("Incomplete CEM round")
        if max(r["b_cost"] for r in elites)>min(r["b_cost"] for r in batch if not r["is_elite"]):
            raise ValueError("Saved elite membership disagrees with latent cost ordering")
        rounds.append(dict(episode=ep,plan_call=call,cem_round=round_number,samples=300,
            successful_prefix_samples=sum(r["prefix_success"] for r in batch),
            successful_prefix_elites=sum(r["prefix_success"] for r in elites),
            minimum_sample_cost=min(r["b_cost"] for r in batch),
            lowest_cost_sample_prefix_success=min(batch,key=lambda r:r["b_cost"])["prefix_success"],
            updated_mean_prefix_success=mean["prefix_success"],updated_mean_cost=mean["b_cost"]))
    final_good=sum(r["prefix_success"] for r in last_elites)
    if not good:pattern="no_successful_prefix_in_captured_pool"
    elif final_good:pattern="successful_prefix_in_final_elites_but_returned_mean_fails"
    else:pattern="successful_prefix_in_pool_but_none_in_final_elites"
    summary=dict(episode=ep,plan_call=call,start_action=control["start_action"],
        candidate_occurrences=len(expanded),successful_prefix_occurrences=len(good),
        all_occurrence_success_pct=100*len(good)/len(expanded),sample_occurrences=len(samples),
        successful_prefix_samples=len(good_samples),sample_success_pct=100*len(good_samples)/len(samples),
        gaussian_samples=len(gaussian),successful_gaussian_prefixes=sum(r["prefix_success"] for r in gaussian),
        unique_first_halves=len(prefix_lookup),successful_unique_first_halves=sum(prefix_lookup.values()),
        unique_first_half_success_pct=100*sum(prefix_lookup.values())/len(prefix_lookup),
        successful_updated_means=len(good)-len(good_samples),
        successful_prefix_samples_ever_elite=sum(r["is_elite"] for r in good_samples),
        final_elites=10,successful_prefix_final_elites=final_good,returned_mean_prefix_success=False,
        lowest_cost_candidate_prefix_success=lowest["prefix_success"],lowest_cost_sample_prefix_success=lowest_sample["prefix_success"],
        lowest_cost_sample_b_cost=lowest_sample["b_cost"],returned_mean_b_cost=float(chosen["b_cost"]),
        returned_mean_full_success=chosen_outcome["full_success"],returned_mean_full_endpoint_success=chosen_outcome["full_endpoint_success"],
        returned_mean_full_endpoint_distance_cm=chosen_outcome["full_endpoint_distance_m"]*100,
        lowest_cost_sample_full_success=lowest_sample["full_success"],
        lowest_cost_sample_full_endpoint_distance_cm=lowest_sample["full_endpoint_distance_cm"],
        best_successful_prefix_b_cost=best_good["b_cost"] if best_good else None,
        best_successful_prefix_cost_minus_returned=best_good["cost_minus_returned_mean"] if best_good else None,
        successful_prefix_samples_better_cost_than_returned=sum(r["b_cost"]<chosen["b_cost"] for r in good_samples),
        successful_prefixes_outside_goal_at_full_endpoint=sum(not r["full_endpoint_success"] for r in good),
        successful_prefixes_inside_goal_at_full_endpoint=len(terminal_good),
        best_successful_prefix_with_goal_terminal_cost=min(r["b_cost"] for r in terminal_good) if terminal_good else None,
        successful_prefix_with_goal_terminal_exists=bool(terminal_good),
        known_old_tail_exact_prefix_occurrences=sum(r["prefix_route_id"]==control["known_tail_prefix_hash"] for r in expanded),
        known_old_tail_first_hit=control["known_tail_first_hit"],
        native_reference_max_error_m=control["native_reference_max_error_m"],observed_selection_pattern=pattern)
    return expanded,rounds,summary


def round_aggregate(rounds):
    result=[]
    for number in range(1,16):
        group=[r for r in rounds if r["cem_round"]==number]
        sampled=sum(r["successful_prefix_samples"] for r in group)
        elite=sum(r["successful_prefix_elites"] for r in group)
        result.append(dict(cem_round=number,pools=len(group),sample_occurrences=300*len(group),
            successful_prefix_samples=sampled,sample_success_pct=100*sampled/(300*len(group)),
            elite_occurrences=10*len(group),successful_prefix_elites=elite,
            elite_success_pct=100*elite/(10*len(group)),
            updated_mean_successes=sum(r["updated_mean_prefix_success"] for r in group),
            mean_minimum_sample_b_cost=float(np.mean([r["minimum_sample_cost"] for r in group])),
            mean_updated_mean_b_cost=float(np.mean([r["updated_mean_cost"] for r in group]))))
    return result


def plot_rounds(rows,output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x=[r["cem_round"] for r in rows]
    with plt.rc_context({"font.size":10,"pdf.fonttype":42}):
        fig,axes=plt.subplots(1,2,figsize=(12,5.2))
        axes[0].plot(x,[r["sample_success_pct"] for r in rows],"o-",color="#0072B2",ms=4,label="300 evaluated samples per pool")
        axes[0].plot(x,[r["elite_success_pct"] for r in rows],"s-",color="#D55E00",ms=4,label="10 selected elites per pool")
        axes[0].set(ylabel="Sequences reaching native success within first 15 actions (%)",title="Successful prefixes during the original CEM search")
        axes[0].set_ylim(bottom=-.1)
        axes[1].plot(x,[r["mean_minimum_sample_b_cost"] for r in rows],"o-",color="#0072B2",ms=4,label="Lowest sample cost")
        axes[1].plot(x,[r["mean_updated_mean_b_cost"] for r in rows],"s-",color="#D55E00",ms=4,label="Updated action-mean cost")
        axes[1].set(ylabel="Full-horizon latent cost (mean across fixed states)",title="Original 30-action endpoint objective")
        for ax in axes:
            ax.set(xlabel="CEM round",xticks=[1,3,5,7,9,11,13,15],xlim=(.7,15.3))
            ax.grid(color=".9",lw=.7)
            ax.spines[["top","right"]].set_visible(False)
            ax.legend(frameon=False,fontsize=9)
        fig.suptitle("B replanning: native prefix success versus the optimized endpoint score",fontsize=14,y=.98)
        fig.text(.5,.015,"47 fixed replanning states in 43 episodes, selected because the previous plan had a successful unexecuted continuation.\n"
                 "Native success: TCP distance <=5 cm. All costs and elite selections are from the original search; no new CEM run.",
                 ha="center",fontsize=8.5,color=".3")
        fig.subplots_adjust(left=.07,right=.99,top=.80,bottom=.25,wspace=.24)
        for suffix in ("png","pdf"):
            fig.savefig(output/f"prefix_success_by_cem_round.{suffix}",dpi=190,bbox_inches="tight")
        plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir",type=Path,default=DEFAULT_RUNS)
    parser.add_argument("--output-dir",type=Path)
    parser.add_argument("--allow-partial",action="store_true")
    args=parser.parse_args()
    runs=args.runs_dir.resolve()
    output=(args.output_dir or runs.parent/"analysis/b_prefix_candidates").resolve()
    db=sqlite3.connect("file:"+(output/"scores.sqlite3").as_posix()+"?mode=ro",uri=True)
    db.execute("BEGIN")
    protocol=json.loads(db.execute("SELECT value FROM metadata WHERE key='protocol'").fetchone()[0])
    pools=db.execute("SELECT episode,plan_call,total,control FROM pools WHERE complete=1 ORDER BY episode,plan_call").fetchall()
    if not args.allow_partial and len(pools)!=47:
        raise ValueError(f"Native replay incomplete: {len(pools)}/47 pools")
    if not pools:raise ValueError("No complete pools to analyze")
    summaries,rounds,successes=[],[],[]
    detailed_path=output/"candidate_prefixes.csv"
    with detailed_path.open("w",newline="",encoding="utf-8") as handle:
        writer=None
        for ep,call,total,control_json in pools:
            scored={key:json.loads(payload) for key,payload in db.execute("SELECT route_id,payload FROM scores WHERE episode=? AND plan_call=?",(ep,call))}
            if len(scored)!=total:raise ValueError("Completed native pool has missing outcomes")
            expanded,by_round,summary=analyze_pool(runs,ep,call,scored,json.loads(control_json))
            if writer is None:
                writer=csv.DictWriter(handle,fieldnames=list(expanded[0]));writer.writeheader()
            writer.writerows(expanded)
            summaries.append(summary);rounds.extend(by_round)
            successes.extend(r for r in expanded if r["prefix_success"])
    db.close()
    csv_file(output/"pool_summary.csv",summaries)
    csv_file(output/"round_summary.csv",rounds)
    csv_file(output/"successful_prefix_candidates.csv",successes,list(expanded[0]))
    patterns={key:sum(r["observed_selection_pattern"]==key for r in summaries) for key in sorted({r["observed_selection_pattern"] for r in summaries})}
    numerator=sum(r["successful_prefix_samples"] for r in summaries)
    denominator=sum(r["sample_occurrences"] for r in summaries)
    aggregate=dict(completed_pools=len(pools),expected_pools=47,complete=len(pools)==47,
        episodes=len({r["episode"] for r in summaries}),sample_occurrences=denominator,successful_prefix_samples=numerator,
        sample_success_pct=100*numerator/denominator,
        candidate_occurrences=sum(r["candidate_occurrences"] for r in summaries),
        successful_prefix_occurrences=sum(r["successful_prefix_occurrences"] for r in summaries),
        successful_updated_means=sum(r["successful_updated_means"] for r in summaries),
        unique_state_and_prefix_pairs=sum(r["unique_first_halves"] for r in summaries),
        successful_unique_state_and_prefix_pairs=sum(r["successful_unique_first_halves"] for r in summaries),
        pools_with_any_successful_prefix=sum(r["successful_prefix_occurrences"]>0 for r in summaries),
        pools_with_a_successful_final_elite=sum(r["successful_prefix_final_elites"]>0 for r in summaries),
        pools_where_all_final_elites_succeed=sum(r["successful_prefix_final_elites"]==10 for r in summaries),
        pools_where_lowest_cost_sample_succeeds=sum(r["lowest_cost_sample_prefix_success"] for r in summaries),
        pools_with_successful_sample_cheaper_than_returned_mean=sum(r["successful_prefix_samples_better_cost_than_returned"]>0 for r in summaries),
        successful_prefix_samples_ever_elite=sum(r["successful_prefix_samples_ever_elite"] for r in summaries),
        successful_prefixes_outside_goal_at_full_endpoint=sum(r["successful_prefixes_outside_goal_at_full_endpoint"] for r in summaries),
        successful_prefixes_inside_goal_at_full_endpoint=sum(r["successful_prefixes_inside_goal_at_full_endpoint"] for r in summaries),
        pools_with_successful_prefix_and_goal_terminal=sum(r["successful_prefix_with_goal_terminal_exists"] for r in summaries),
        pools_with_goal_terminal_successful_prefix_but_returned_terminal_outside=sum(r["successful_prefix_with_goal_terminal_exists"] and not r["returned_mean_full_endpoint_success"] for r in summaries),
        known_old_tail_exact_prefix_occurrences=sum(r["known_old_tail_exact_prefix_occurrences"] for r in summaries),
        returned_mean_prefix_successes=0,known_old_tail_successes=len(pools),
        native_reference_max_error_m=max(r["native_reference_max_error_m"] for r in summaries),selection_patterns=patterns)
    write_json(output/"summary.json",aggregate)
    round_totals=round_aggregate(rounds)
    csv_file(output/"cem_round_aggregate.csv",round_totals)
    if len(pools)==47:plot_rounds(round_totals,output)
    lines=["# B's newly proposed first halves after a successful continuation", "",
        f"Coverage: {len(pools)}/47 replanning states. "+("Complete." if len(pools)==47 else "PARTIAL; completed pools may be unrepresentative."), "",
        "Each state follows a selected B plan that would succeed beyond its executed prefix. The previous 15-action remainder is replayed from the new state and must still succeed. The current chosen mean and previous full-plan trace must reproduce saved native outcomes. No candidates are added or changed.", "",
        "A prefix succeeds if native TCP distance is <=5 cm at any of its first 15 raw actions. Original B cost ranks the complete 30-action endpoint. All costs and elite memberships below are those saved during the original search.", "",
        f"Successful evaluated sample prefixes: {numerator}/{denominator} ({aggregate['sample_success_pct']:.4f}%). There are 300 evaluated samples per round (299 fresh Gaussian samples plus the included distribution mean), 15 rounds per pool. Fifteen additional updated means are reported separately.", "",
        f"Pools containing any successful captured prefix: {aggregate['pools_with_any_successful_prefix']}/{len(pools)}. Returned final means with successful prefixes: 0. Successful known previous remainders: {len(pools)}/{len(pools)}.", "",
        "## Selection evidence", "",
        f"Pools with a successful prefix among the ten final-round elites: {aggregate['pools_with_a_successful_final_elite']}. Pools where the lowest-cost sampled sequence has a successful prefix: {aggregate['pools_where_lowest_cost_sample_succeeds']}. Pools with a successful sampled prefix whose full-horizon latent cost is lower than the returned mean: {aggregate['pools_with_successful_sample_cheaper_than_returned_mean']}.", "",
        f"Successful sampled prefixes selected as an elite in their own round: {aggregate['successful_prefix_samples_ever_elite']}/{numerator}. Successful captured prefixes whose diagnostic full-horizon endpoint lies outside the goal region: {aggregate['successful_prefixes_outside_goal_at_full_endpoint']}/{aggregate['successful_prefix_occurrences']}.", "",
        "CEM returns the final elite action mean, not its best sample. Elite membership can therefore coexist with a failed returned mean. Endpoint ranking can also prefer a failed prefix. These observations describe selection behavior, not a causal attribution to representation information alone.", "",
        f"Successful captured prefixes that also end inside the goal region at action 30: {aggregate['successful_prefixes_inside_goal_at_full_endpoint']}. In {aggregate['pools_with_goal_terminal_successful_prefix_but_returned_terminal_outside']} pools such an option exists while the returned mean's action-30 endpoint lies outside the goal region. These are endpoint-matched native checks; they do not isolate visual geometry from goal-image/proprioceptive objective choices.", "",
        "| Episode | New planning starts at action | Successful sample prefixes / 4500 | Rate (%) | Successful final elites / 10 | Best successful-prefix cost minus returned-mean cost | Pattern |",
        "|---:|---:|---:|---:|---:|---:|---|"]
    for r in summaries:
        gap=r["best_successful_prefix_cost_minus_returned"]
        lines.append(f"| {r['episode']} | {r['start_action']} | {r['successful_prefix_samples']} | {r['sample_success_pct']:.4f} | {r['successful_prefix_final_elites']} | {gap:.6f} | {r['observed_selection_pattern']} |" if gap is not None else
                     f"| {r['episode']} | {r['start_action']} | 0 | 0 | 0 | unavailable: no successful prefix | {r['observed_selection_pattern']} |")
    lines += ["", "## Scope and interpretation", "",
        "- This is an outcome-conditioned set of 47 decisions in 43 B episodes. It is not a representative success rate for every B search state or independent candidate trials.",
        "- `candidate_prefixes.csv` retains all occurrences. `pool_summary.csv` also reports distinct first-half routes so repeated included means do not inflate coverage claims.",
        "- Native scoring is cached once per identical full raw action sequence. Prefix success is measured for every saved occurrence. Full native continuation is measured for every successful prefix, elite, updated mean, and per-round minimum sample; other full physical outcomes remain explicitly unavailable.",
        "- No successful prefix in the captured pool is a conditional coverage failure: a known feasible remainder exists at that state, but this captured pool did not contain a successful prefix. It is not a claim about all admissible actions.",
        "- Successful prefixes in elites with a failed final mean show that averaging does not preserve their native feasibility. Some successful members do not establish that all members succeed.",
        "- A successful prefix with a higher full-horizon latent cost can be rejected by the original score. This does not separate endpoint timing, visual/proprioceptive weights, representation geometry, or finite search.",
        "- The previous remainder is checked as a feasibility reference only. It is not inserted into CEM or assigned an invented 30-action latent score.",
        "- Exact absence of an old prefix alone has little meaning in a continuous action space; native success coverage is the substantive check.", ""]
    (output/"results.md").write_text("\n".join(lines),encoding="utf-8")
    write_json(output/"analysis_protocol.json",dict(source_replay_protocol=protocol,complete=len(pools)==47,
        analysis="Original full-horizon latent costs and native prefix outcomes; no score changes or new search",
        native_success_threshold_m=.05,denominator="4500 CEM sample occurrences per pool, including 15 included means; separate 15 updated means",
        selected_states=[dict(episode=r["episode"],plan_call=r["plan_call"],start_action=r["start_action"]) for r in summaries]))
    print(json.dumps(aggregate,indent=2),flush=True)
    print(f"Results: {output/'results.md'}",flush=True)


if __name__=="__main__":main()

"""Compare the executed prefix and full horizon of A/B's selected plans.

Run in conda AI from jepa-wms:
    python -m evals.simu_env_planning.planning.analyze_reach_wall_prefix

A uses its completed simulator replay database, not its predicted embedding.
B uses its own saved oracle rollout. Only the deployed final CEM means are
included. No new simulator, search, encoder, or predictive model is run.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sqlite3

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch

from .plot_reach_wall_distance import DEFAULT_RUNS, METHODS, write_csv


EPISODES = 48
STARTS = (0, 15, 30, 45, 60, 75, 90)
A_FOLDER = "bounded_a_i15_n300_ep48"


def file_hash(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def close(actual, expected, label):
    if not math.isclose(float(actual), float(expected), abs_tol=1e-8, rel_tol=0):
        raise ValueError(f"{label}: {actual} != {expected}")


def executions(runs, method, ep, hashes):
    path = runs / METHODS[method]["folder"] / f"ep_{ep}/execution.csv"
    hashes[str(path)] = file_hash(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if [int(row["simulator_step"]) for row in rows] != list(range(1, len(rows)+1)):
        raise ValueError("Noncontiguous execution")
    if len(rows) not in (99, 100) or any(row["success"] == "True" for row in rows[:-1]):
        raise ValueError("Unexpected actual stopping behavior")
    for row in rows:
        if (row["success"] == "True") != (float(row["tcp_goal_distance_m"]) <= .05):
            raise ValueError("Native success flag disagrees with distance")
    return rows


def compact_record(method, ep, call, metrics, actual, all_actual, latent_cost, source):
    """Keep raw horizon endpoints distinct from stop-on-success outcomes."""
    start = int(actual[0]["simulator_step"])-1
    horizon = int(metrics["full_horizon_steps"])
    prefix = int(metrics["prefix_steps"])
    hit = metrics["task_first_hit_step"]
    if start != STARTS[call] or horizon != min(30, 100-start) or prefix != min(15, horizon):
        raise ValueError("Unexpected start/horizon/prefix schedule")
    success = bool(metrics["task_success"])
    prefix_success = bool(metrics["prefix_success"])
    if success != (hit is not None) or hit is not None and not 0 < hit <= horizon:
        raise ValueError("Full-plan first-hit fields disagree")
    if prefix_success != (hit is not None and hit <= prefix):
        raise ValueError("Prefix first-hit fields disagree")
    expected_execution = hit if prefix_success else prefix
    if len(actual) != expected_execution or metrics["prefix_task_steps"] != expected_execution:
        raise ValueError("Actual execution differs from first-hit stopping rule")
    if (actual[-1]["success"] == "True") != prefix_success:
        raise ValueError("Saved prefix success differs from actual execution")
    close(metrics["prefix_task_end_distance_m"], actual[-1]["tcp_goal_distance_m"], "Actual prefix endpoint")
    if len(actual) == prefix:
        close(metrics["prefix_horizon_goal_distance_m"], actual[-1]["tcp_goal_distance_m"], "Raw prefix endpoint")
    if bool(metrics["full_horizon_success"]) != (metrics["full_horizon_goal_distance_m"] <= .05):
        raise ValueError("Full terminal-success flag disagrees with distance")
    if success != (metrics["task_end_distance_m"] <= .05):
        raise ValueError("Stopped full-plan outcome disagrees with first-hit success")
    if metrics["task_steps"] != (hit if success else horizon):
        raise ValueError("Stopped full-plan action count disagrees")
    numbers = [metrics[name] for name in ("task_end_distance_m", "prefix_task_end_distance_m",
               "full_horizon_goal_distance_m", "prefix_horizon_goal_distance_m")]
    if any(not math.isfinite(value) or value < 0 for value in numbers):
        raise ValueError("Invalid native distance")
    start_cm = float(all_actual[start-1]["tcp_goal_distance_m"])*100 if start else None
    prefix_cm = float(metrics["prefix_task_end_distance_m"])*100
    full_cm = float(metrics["task_end_distance_m"])*100
    return dict(method=method, episode=ep, plan_call=call, start_action=start,
        full_horizon_actions=horizon, prefix_limit_actions=prefix, actual_executed_actions=len(actual),
        start_distance_cm=start_cm, prefix_success=prefix_success, full_success=success,
        first_hit_action_in_plan=hit, first_hit_episode_action=start+hit if hit is not None else None,
        success_only_after_prefix=success and not prefix_success,
        prefix_stopped_distance_cm=prefix_cm, full_stopped_distance_cm=full_cm,
        prefix_raw_endpoint_distance_cm=float(metrics["prefix_horizon_goal_distance_m"])*100,
        full_raw_endpoint_distance_cm=float(metrics["full_horizon_goal_distance_m"])*100,
        full_raw_endpoint_success=bool(metrics["full_horizon_success"]),
        full_closer_than_prefix=full_cm < prefix_cm-1e-6,
        suffix_distance_improvement_cm=prefix_cm-full_cm,
        prefix_distance_improvement_cm=start_cm-prefix_cm if start_cm is not None else None,
        total_distance_improvement_cm=start_cm-full_cm if start_cm is not None else None,
        optimization_endpoint_latent_cost=float(latent_cost), source=source)


def extract_a(runs, hashes):
    run = runs / A_FOLDER
    database = run / "branch_b/scores.sqlite3"
    query = """SELECT s.episode,s.plan_call,s.candidate_index,s.payload,p.total,p.source_sha256
        FROM pools p JOIN scores s ON s.episode=p.episode AND s.plan_call=p.plan_call
        AND s.candidate_index=p.total-1 ORDER BY p.episode,p.plan_call"""
    rows, payload_hashes, bundle_hashes = [], {}, {}
    with sqlite3.connect("file:"+database.as_posix()+"?mode=ro", uri=True) as connection:
        connection.execute("BEGIN")
        raw = connection.execute(query).fetchall()
        if len(raw) != EPISODES*7:
            raise ValueError("Expected 336 complete A decision pools")
        for ep in range(EPISODES):
            actual = executions(runs, "A", ep, hashes)
            # Verify identity of the deployed mean against A's independent log.
            log_path = run / f"mw-reach-wall/ep_{ep}/branch_a.csv"
            hashes[str(log_path)] = file_hash(log_path)
            with log_path.open(newline="", encoding="utf-8-sig") as handle:
                deployed = {int(r["plan_call"]): r for r in csv.DictReader(handle) if r["is_deployed_output"] == "True"}
            if set(deployed) != set(range(7)):
                raise ValueError("A's deployed-mean log is incomplete")
            for source_ep, call, index, payload, total, bundle_sha in raw[ep*7:(ep+1)*7]:
                saved = json.loads(payload)
                if source_ep != ep or total != 4515 or index != 4514 or not saved["is_deployed_output"]:
                    raise ValueError("Selected A record is not its actual final mean")
                if saved["candidate"] != "mean_round_15" or saved["origin"] != "updated_mean":
                    raise ValueError("Unexpected A candidate identity")
                if deployed[call]["candidate"] != saved["candidate"]:
                    raise ValueError("A candidate identity disagrees with source log")
                close(saved["a_cost"], float(deployed[call]["a_cost"]), "A deployed mean cost")
                batch = [r for r in actual if int(r["plan_call"]) == call]
                row = compact_record("A", ep, call, saved, batch, actual, saved["a_cost"],
                                     "A selected mean: completed simulator replay database")
                rows.append(row)
                payload_hashes[f"{ep}/{call}"] = hashlib.sha256(payload.encode()).hexdigest()
                bundle_hashes[f"{ep}/{call}"] = bundle_sha
        producer = json.loads(connection.execute("SELECT value FROM metadata WHERE key='protocol'").fetchone()[0])
    return rows, dict(database=str(database), database_size_bytes=database.stat().st_size,
        query=query, selected_payload_sha256=payload_hashes, replay_source_bundle_sha256=bundle_hashes,
        producer_protocol_sha256=hashlib.sha256(json.dumps(producer,sort_keys=True).encode()).hexdigest())


def extract_b(runs, hashes):
    rows, traces = [], []
    for ep in range(EPISODES):
        actual = executions(runs, "B", ep, hashes)
        for call in range(7):
            path = runs / f"closed_loop_b/ep_{ep}/plan_{call:03d}.pt"
            hashes[str(path)] = file_hash(path)
            saved = torch.load(path, map_location="cpu", weights_only=False)
            if saved["episode"] != ep or saved["plan_call"] != call or saved["branch"] != "B_closed_loop":
                raise ValueError("B saved decision identity disagrees")
            means = [c for c in saved["candidates"] if c["is_deployed_output"]]
            if len(means) != 1 or means[0]["name"] != "mean_round_15" or len(saved["candidates"]) != 4515:
                raise ValueError("B returned final mean is ambiguous")
            mean = means[0]
            metrics, trace = saved["final_mean_rollout"]["metrics"], saved["final_mean_rollout"]["trace"]
            if len(trace) != metrics["full_horizon_steps"] or len(trace) != len(mean["actions_simulator"]):
                raise ValueError("B selected rollout has the wrong length")
            hits = []
            for step, point in enumerate(trace, 1):
                if bool(point["success"]) != (point["distance"] <= .05) or not math.isfinite(point["distance"]):
                    raise ValueError("B native trace has an invalid success/distance")
                if point["success"]:
                    hits.append(step)
                traces.append(dict(method="B", episode=ep, plan_call=call, action_in_plan=step,
                    episode_action=saved["decision"]["episode_steps_elapsed"]+step,
                    distance_cm=float(point["distance"])*100, native_success=bool(point["success"]),
                    inside_executed_prefix=step <= metrics["prefix_steps"]))
            if metrics["task_first_hit_step"] != (hits[0] if hits else None):
                raise ValueError("B metrics do not describe the saved native trace")
            close(trace[-1]["distance"], metrics["full_horizon_goal_distance_m"], "B full endpoint")
            close(trace[metrics["prefix_steps"]-1]["distance"], metrics["prefix_horizon_goal_distance_m"], "B raw prefix endpoint")
            if hits:
                close(trace[hits[0]-1]["distance"], metrics["task_end_distance_m"], "B first-hit endpoint")
            batch = [r for r in actual if int(r["plan_call"]) == call]
            for index, record in enumerate(batch):
                close(trace[index]["distance"], record["tcp_goal_distance_m"], "B actual-prefix distance")
                if trace[index]["success"] != (record["success"] == "True"):
                    raise ValueError("B actual-prefix success changed")
                recorded_action = np.array([float(record[f"action_{d}"]) for d in range(4)])
                np.testing.assert_allclose(mean["actions_simulator"][index].numpy(),recorded_action,atol=1e-6,rtol=0)
            rows.append(compact_record("B",ep,call,metrics,batch,actual,mean["b_cost"],"B own selected mean: saved native oracle rollout"))
            del saved
        if (ep+1) % 8 == 0:
            print(f"B selected-plan extraction and execution checks: {ep+1}/48 episodes",flush=True)
    return rows, traces


def summarize(rows, **keys):
    """Counts describe related planning calls, not independent episode trials."""
    distances = np.array([r["full_stopped_distance_cm"] for r in rows])
    prefixes = np.array([r["prefix_stopped_distance_cm"] for r in rows])
    hits = [r["first_hit_action_in_plan"] for r in rows if r["full_success"]]
    return dict(**keys, plans=len(rows), episodes=len({r["episode"] for r in rows}),
        prefix_successes=sum(r["prefix_success"] for r in rows), full_successes=sum(r["full_success"] for r in rows),
        success_only_after_prefix=sum(r["success_only_after_prefix"] for r in rows),
        episodes_with_success_only_after_prefix=len({r["episode"] for r in rows if r["success_only_after_prefix"]}),
        full_raw_endpoint_successes=sum(r["full_raw_endpoint_success"] for r in rows),
        full_closer_than_prefix=sum(r["full_closer_than_prefix"] for r in rows),
        mean_prefix_stopped_distance_cm=float(prefixes.mean()), mean_full_stopped_distance_cm=float(distances.mean()),
        median_prefix_stopped_distance_cm=float(np.median(prefixes)), median_full_stopped_distance_cm=float(np.median(distances)),
        mean_suffix_distance_improvement_cm=float((prefixes-distances).mean()),
        mean_first_hit_action_in_plan=float(np.mean(hits)) if hits else None)


def episode_summary(rows):
    result = []
    for method in ("A", "B"):
        for ep in range(EPISODES):
            batch = sorted((r for r in rows if r["method"] == method and r["episode"] == ep),key=lambda r:r["plan_call"])
            run = longest = 0
            for row in batch:
                run = run+1 if row["success_only_after_prefix"] else 0
                longest = max(longest,run)
            result.append(dict(method=method,episode=ep,planning_calls=len(batch),
                actual_episode_success=any(r["prefix_success"] for r in batch),
                full_plan_success_calls=sum(r["full_success"] for r in batch),
                future_only_success_calls=sum(r["success_only_after_prefix"] for r in batch),
                consecutive_future_only_success_calls=longest,
                mean_suffix_improvement_30_action_plans_cm=float(np.mean([r["suffix_distance_improvement_cm"] for r in batch if r["full_horizon_actions"] == 30]))))
    return result


def plot_distances(rows, output):
    with plt.rc_context({"font.size":10,"pdf.fonttype":42}):
        fig,axes = plt.subplots(1,2,figsize=(12.6,5.5),sharey=True)
        for ax,method in zip(axes,("A","B")):
            color = METHODS[method]["color"]
            for field,style,marker in (("prefix_stopped_distance_cm","--","o"),("full_stopped_distance_cm","-","s")):
                values = np.array([[r[field] for r in rows if r["method"] == method and r["plan_call"] == call] for call in range(7)])
                if values.shape != (7,48):
                    raise ValueError("Plot does not have 48 trials per planning call")
                q1,median,q3 = np.quantile(values,[.25,.5,.75],axis=1)
                ax.plot(range(7),median,color=color,ls=style,marker=marker,ms=5,lw=1.7)
                ax.fill_between(range(7),q1,q3,color=color,alpha=.10 if style == "--" else .18)
            ax.axhline(5,color="#A12A32",ls=":",lw=1.2)
            ax.axvspan(4.5,6.4,color=".92",zorder=0)
            ax.set(title=f"{method}: own returned final-mean plans",xlim=(-.2,6.3),ylim=(3,24),
                   xticks=range(7),xticklabels=["0\n15 / 30","15\n15 / 30","30\n15 / 30","45\n15 / 30","60\n15 / 30","75\n15 / 25","90\n10 / 10"],
                   xlabel="Planning starts at raw action\nPrefix limit / full horizon (raw actions)")
            ax.grid(axis="y",color=".88",lw=.7)
            ax.spines[["top","right"]].set_visible(False)
        axes[0].set_ylabel("Native TCP-to-goal distance (cm)")
        handles = [Line2D([],[],color=".3",ls="--",marker="o",label="Prefix outcome"),
                   Line2D([],[],color=".3",ls="-",marker="s",label="Full-plan outcome"),
                   Line2D([],[],color="#A12A32",ls=":",label="5 cm success threshold")]
        fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.5,.085),ncol=3,frameon=False)
        fig.suptitle("Selected plan: executed prefix versus full simulator rollout",fontsize=15,y=.98)
        fig.text(.5,.015,"Median and 25th-75th percentiles across 48 trials. Each outcome stops at first success or its action limit.\n"
                 "Full plans are counterfactual after the executed prefix. A/B use their own visited states. Gray area: shortened horizons.",
                 ha="center",fontsize=8.5,color=".3")
        fig.subplots_adjust(left=.07,right=.99,top=.83,bottom=.32,wspace=.14)
        for suffix in ("png","pdf"):
            fig.savefig(output/f"prefix_vs_full_distance.{suffix}",dpi=200,bbox_inches="tight")
        plt.close(fig)


def report(output, call_rows, horizon_rows, episodes):
    text = ["# Reach-Wall selected plans: prefix versus full horizon", "",
        "Both A and B use native simulator outcomes here. A's learned predictions are not being interpreted as task-success predictions. A is evaluated using its completed fixed-pool replay; B uses its own saved oracle rollouts. Only each controller's actual returned final elite mean is included.", "",
        "Native success means TCP-to-goal distance <=5 cm at any action. Distance outcomes stop at first success, otherwise at the relevant prefix/horizon limit. Raw terminal distances are retained separately in the detailed data.", "",
        "## Standard horizons: 15-action prefix versus full 30-action plan", "",
        "| Controller | Plans | Prefix successes | Full-plan successes | Success only after prefix | Episodes with a future-only success | Mean prefix distance (cm) | Mean full distance (cm) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in horizon_rows:
        if row["full_horizon_actions"] == 30:
            text.append(f"| {row['method']} | {row['plans']} | {row['prefix_successes']} | {row['full_successes']} | {row['success_only_after_prefix']} | {row['episodes_with_success_only_after_prefix']} | {row['mean_prefix_stopped_distance_cm']:.2f} | {row['mean_full_stopped_distance_cm']:.2f} |")
    text += ["", "## Every planning call", "",
        "The initial call starts at the same physical state across A/B. Their later states differ, so these rows are within-controller descriptions, not controlled cross-controller search comparisons.", "",
        "| Controller | Start action | Prefix / horizon | Plans | Prefix successes | Full-plan successes | Mean prefix distance (cm) | Mean full distance (cm) | Full closer than prefix |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for r in call_rows:
        text.append(f"| {r['method']} | {r['start_action']} | {r['prefix_limit_actions']} / {r['full_horizon_actions']} | {r['plans']} | {r['prefix_successes']} | {r['full_successes']} | {r['mean_prefix_stopped_distance_cm']:.2f} | {r['mean_full_stopped_distance_cm']:.2f} | {r['full_closer_than_prefix']} |")
    text += ["", "## Repeated future-only success", ""]
    for method in ("A","B"):
        batch = [r for r in episodes if r["method"] == method]
        text.append(f"{method}: {sum(r['future_only_success_calls'] > 0 for r in batch)}/48 episodes contain at least one selected full plan that succeeds beyond its prefix; {sum(r['consecutive_future_only_success_calls'] >= 2 for r in batch)}/48 contain consecutive such calls. Maximum consecutive calls: {max(r['consecutive_future_only_success_calls'] for r in batch)}.")
    text += ["", "## Definitions and limits", "",
        "- These are counts of related planning calls in 48 episodes per controller, not hundreds of independent episode trials.",
        "- Succeeding only after a prefix can be normal in receding-horizon control. It is not by itself proof of harmful postponement.",
        "- Full-plan outcomes after the prefix are counterfactual: they describe completing that exact saved action sequence without replanning.",
        "- At start action 75, compare 15 versus 25 actions. At 90 compare a maximum 10-action prefix with the same 10-action full plan. These calls are excluded from the primary 15-versus-30 aggregate.",
        "- A episode 46 stops at action 99: its final planned prefix limit is ten, but native success stops actual execution after nine. Stop-on-success distance is used, and the raw horizon endpoint remains separately labelled.",
        "- Start-action-zero native distance was not logged. Start-distance and prefix-progress fields are blank for that call; no hand-coordinate proxy is substituted.",
        "- A does not have per-action counterfactual distances in the replay database. Its first-hit, prefix and full-endpoint measurements suffice for this comparison. B's complete selected-plan traces are also exported.",
        "- The comparison does not establish the cause of late improvement or assess all candidate pools. No model, encoder, new search, or simulator replay was run.", ""]
    (output/"results.md").write_text("\n".join(text),encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir",type=Path,default=DEFAULT_RUNS)
    parser.add_argument("--output-dir",type=Path)
    args = parser.parse_args()
    runs = args.runs_dir.resolve()
    output = (args.output_dir or runs.parent/"analysis/prefix_vs_full").resolve()
    output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    hashes = {}
    a,provenance = extract_a(runs,hashes)
    print("A selected-plan extraction and native endpoint checks: 336/336",flush=True)
    b,traces = extract_b(runs,hashes)
    rows = a+b
    if len({(r["method"],r["episode"],r["plan_call"]) for r in rows}) != 672:
        raise ValueError("Missing or duplicate selected plans")
    call_rows, horizon_rows = [], []
    for method in ("A","B"):
        for call,start in enumerate(STARTS):
            batch = [r for r in rows if r["method"] == method and r["plan_call"] == call]
            call_rows.append(summarize(batch,method=method,plan_call=call,start_action=start,
                full_horizon_actions=batch[0]["full_horizon_actions"],prefix_limit_actions=batch[0]["prefix_limit_actions"]))
        for horizon in (30,25,10):
            batch = [r for r in rows if r["method"] == method and r["full_horizon_actions"] == horizon]
            horizon_rows.append(summarize(batch,method=method,full_horizon_actions=horizon,
                prefix_limit_actions=batch[0]["prefix_limit_actions"]))
    episodes = episode_summary(rows)
    lookup = {(r["method"],r["episode"],r["plan_call"]):r for r in rows}
    promises = []
    for row in rows:
        if not row["success_only_after_prefix"]:
            continue
        later = lookup.get((row["method"],row["episode"],row["plan_call"]+1))
        promises.append(dict(method=row["method"],episode=row["episode"],plan_call=row["plan_call"],
            start_action=row["start_action"],first_hit_action_in_plan=row["first_hit_action_in_plan"],
            first_hit_episode_action=row["first_hit_episode_action"],
            next_plan_full_success=later["full_success"] if later else None,
            next_plan_first_hit_episode_action=later["first_hit_episode_action"] if later else None,
            next_first_hit_shift_actions=later["first_hit_episode_action"]-row["first_hit_episode_action"] if later and later["full_success"] else None))
    for filename,batch in (("selected_plans.csv",rows),("planning_call_summary.csv",call_rows),
                           ("horizon_summary.csv",horizon_rows),("episode_summary.csv",episodes),
                           ("b_selected_rollout_traces.csv",traces)):
        write_csv(output/filename,batch)
    if promises:
        write_csv(output/"future_only_successes.csv",promises)
    summary = dict(episodes_per_controller=48,selected_plans=len(rows),primary_30_action_horizon=[r for r in horizon_rows if r["full_horizon_actions"] == 30],
        shortened_horizons=[r for r in horizon_rows if r["full_horizon_actions"] != 30],
        actual_a_episode_successes=sum(r["actual_episode_success"] for r in episodes if r["method"] == "A"),
        actual_b_episode_successes=sum(r["actual_episode_success"] for r in episodes if r["method"] == "B"))
    (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    protocol = dict(source_root=str(runs),source_file_sha256=hashes,a_replay_database=provenance,
        selected_candidate="Actual deployed mean_round_15, not minimum sample or reranked A candidates",
        native_success_threshold_m=.05,primary_horizon=30,primary_prefix_limit=15,
        distance_outcome="First-hit distance when successful; otherwise distance at prefix/full limit",
        raw_endpoints="Also retained as separate fields; diagnostic continuation can leave the success region",
        temporal_unit="raw simulator action",missing_initial_distance="Not recorded at action zero; leave blank",
        a_and_b_states="Each uses its own visited state; only the initial decision states are paired",
        no_new_simulation=True,no_model_inference=True,no_new_search=True)
    (output/"protocol.json").write_text(json.dumps(protocol,indent=2)+"\n",encoding="utf-8")
    report(output,call_rows,horizon_rows,episodes)
    plot_distances(rows,output)
    print(json.dumps(summary,indent=2),flush=True)
    print(f"Results: {output}",flush=True)


if __name__ == "__main__":
    main()

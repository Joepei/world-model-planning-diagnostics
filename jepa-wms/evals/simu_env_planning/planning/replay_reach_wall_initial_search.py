"""Native evaluation of B's independent initial candidate pools, paired to A.

The first decision has matched physics/goal, horizon, proposal budget, and first
population. Reuse completed A replay measurements; replay all B candidates over
30 raw actions. No new search, encoder run, or controller intervention.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import csv
import ctypes
import json
import os
from pathlib import Path
import sqlite3
from time import perf_counter

import numpy as np
import psutil
import torch

from . import replay_reach_wall_prefix_candidates as native


DEFAULT_RUNS=native.DEFAULT_RUNS


def a_metrics_match(outcome,row):
    if outcome["full_first_hit"]!=row["task_first_hit_step"] or outcome["prefix_success"]!=row["prefix_success"]:
        raise ValueError("Paired native first-hit/prefix measurements disagree")
    errors=[abs(outcome["full_endpoint_distance_m"]-row["full_horizon_goal_distance_m"]),
        abs(outcome["prefix_endpoint_distance_m"]-row["prefix_horizon_goal_distance_m"]),
        abs(outcome["prefix_stopped_distance_m"]-row["prefix_task_end_distance_m"])]
    hit=outcome["full_first_hit"]
    stopped=outcome["distance_trace_m"][hit-1] if hit else outcome["distance_trace_m"][-1]
    errors.append(abs(stopped-row["task_end_distance_m"]))
    if max(errors)>1e-8:
        raise ValueError(f"Paired native distance measurements disagree: {max(errors)}")
    return max(errors)


def score_episode(runs_string,database_string,episode):
    runs,database=Path(runs_string),Path(database_string)
    run=runs/"bounded_a_i15_n300_ep48"
    a_path=run/f"mw-reach-wall/ep_{episode}/branch_a/plan_000.pt"
    b_path=runs/f"closed_loop_b/ep_{episode}/plan_000.pt"
    a_sha,b_sha=native.sha(a_path),native.sha(b_path)
    db=native.connect(database)
    old=db.execute("SELECT a_sha,b_sha,complete,control FROM pools WHERE episode=?",(episode,)).fetchone()
    if old and old[:2]!=(a_sha,b_sha):
        raise ValueError("An original candidate pool changed; use another analysis directory")
    if old and old[2]:
        db.close()
        return dict(episode=episode,cached=True,**json.loads(old[3])["counts"])
    a=torch.load(a_path,map_location="cpu",weights_only=False)
    b=torch.load(b_path,map_location="cpu",weights_only=False)
    ac,bc=a["candidates"],b["candidates"]
    if len(ac)!=4515 or len(bc)!=4515 or a["episode_steps_elapsed"]!=0 or b["decision"]["episode_steps_elapsed"]!=0:
        raise ValueError("Expected full first-call captures at episode action zero")
    for key in ("integration_state","model_arrays","history","np_random_state","last_state"):
        np.testing.assert_equal(a["simulator_snapshot"][key],b["decision"]["simulator_snapshot"][key])
    for key in a["goal_encoding"]:
        torch.testing.assert_close(a["goal_encoding"][key],b["goal_encoding"][key],rtol=0,atol=0)
    for x,y in zip(ac[:300],bc[:300]):
        if x["cem_round"]!=1 or y["cem_round"]!=1 or not torch.equal(x["actions_normalized"],y["actions_normalized"]):
            raise ValueError("The first population is not exactly paired")
        if not torch.equal(x["actions_simulator"],y["actions_simulator"]):
            raise ValueError("Paired initial samples have different physical commands")
    with sqlite3.connect("file:"+(run/"branch_b/scores.sqlite3").as_posix()+"?mode=ro",uri=True) as source:
        pool=source.execute("SELECT source_sha256,total FROM pools WHERE episode=? AND plan_call=0",(episode,)).fetchone()
        if pool!=(a_sha,4515):
            raise ValueError("A replay scores do not refer to this initial candidate pool")
        a_rows=[json.loads(r[0]) for r in source.execute("SELECT payload FROM scores WHERE episode=? AND plan_call=0 ORDER BY candidate_index",(episode,))]
    if len(a_rows)!=4515:
        raise ValueError("Incomplete A native replay")
    a_lookup={r["route_id"]:r for r in a_rows}
    for c,r in zip(ac,a_rows):
        if native.route_id(c["actions_normalized"].numpy())!=r["route_id"] or c["name"]!=r["candidate"]:
            raise ValueError("A candidate identity disagrees with its replay database")
    decision=b["decision"]
    native.restore(decision)
    initial_distance=float(np.linalg.norm(native.ENV.unwrapped.tcp_center-native.ENV.unwrapped._target_pos))
    if initial_distance<=.05:
        raise ValueError("Unexpected initially successful trial")
    selected=next(c for c in bc if c["is_deployed_output"])
    selected_check=native.rollout(decision,selected["actions_simulator"].numpy(),True)
    selected_error=native.check_trace(selected_check,b["final_mean_rollout"]["trace"])
    with (b_path.parent/"execution.csv").open(newline="") as handle:
        actual=[r for r in csv.DictReader(handle) if int(r["plan_call"])==0]
    np.testing.assert_allclose(selected_check["distance_trace_m"][:15],
        [float(r["tcp_goal_distance_m"]) for r in actual],atol=1e-8,rtol=0)
    routes={}
    for c in bc:
        raw=c["actions_simulator"].numpy()
        key=native.route_id(raw)
        normal=native.route_id(c["actions_normalized"].numpy())
        if key not in routes:routes[key]=dict(actions=raw,normalized_ids=set())
        routes[key]["normalized_ids"].add(normal)
    if len(routes)!=4501:
        raise ValueError("Unexpected unique-route count")
    with db:db.execute("INSERT OR IGNORE INTO pools VALUES (?,?,?,?,?,?,?)",(episode,a_sha,b_sha,len(routes),0,None,None))
    outcomes={key:json.loads(payload) for key,payload in db.execute("SELECT route_id,payload FROM scores WHERE episode=?",(episode,))}
    selected_key=native.route_id(selected["actions_simulator"].numpy())
    pending=[]
    shared=set()
    reference_error=0.
    started=perf_counter()
    for key,route in routes.items():
        if key not in outcomes:
            outcome=selected_check if key==selected_key else native.rollout(decision,route["actions"],True)
            outcomes[key]=outcome
            pending.append((episode,key,json.dumps(outcome)))
        outcome=outcomes[key]
        if not outcome["full_outcome_available"] or outcome["simulated_actions"]!=30:
            raise ValueError("Every B route must have a complete native 30-action outcome")
        for normal in route["normalized_ids"]:
            if normal in a_lookup:
                reference_error=max(reference_error,a_metrics_match(outcome,a_lookup[normal]))
                shared.add(normal)
        if len(pending)>=100:
            with db:db.executemany("INSERT INTO scores VALUES (?,?,?)",pending)
            pending=[]
    if pending:
        with db:db.executemany("INSERT INTO scores VALUES (?,?,?)",pending)
    if len(outcomes)!=4501 or len(shared)<300:
        raise ValueError("Missing scored routes or shared initial-population controls")
    counts=dict(unique_b_routes=len(routes),b_full_successes=sum(r["full_success"] for r in outcomes.values()),
        b_prefix_successes=sum(r["prefix_success"] for r in outcomes.values()),shared_native_routes_checked=len(shared),
        native_scoring_actions=sum(r["simulated_actions"] for r in outcomes.values()))
    control=dict(episode=episode,a_source=str(a_path),b_source=str(b_path),a_sha256=a_sha,b_sha256=b_sha,
        paired_initial_physics=True,paired_goal_encoding=True,paired_first_population=300,
        shared_native_max_error_m=reference_error,selected_native_max_error_m=selected_error,
        initial_tcp_goal_distance_m=initial_distance,counts=counts,worker_seconds=perf_counter()-started)
    with db:db.execute("UPDATE pools SET complete=1,source=?,control=? WHERE episode=?",(json.dumps(dict(a_source=str(a_path),b_source=str(b_path))),json.dumps(control),episode))
    db.close()
    return dict(episode=episode,cached=False,**counts)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir",type=Path,default=DEFAULT_RUNS)
    parser.add_argument("--output-dir",type=Path)
    parser.add_argument("--workers",type=int,choices=range(1,5),default=3)
    parser.add_argument("--max-pools",type=int)
    args=parser.parse_args()
    runs=args.runs_dir.resolve()
    output=(args.output_dir or runs.parent/"analysis/initial_search_ab").resolve()
    output.mkdir(parents=True,exist_ok=True)
    config=runs/"bounded_a_i15_n300_ep48/args_eval.yaml"
    b_protocol_path=runs/"closed_loop_b/protocol.json"
    b_protocol=json.loads(b_protocol_path.read_text())
    if b_protocol["source_args_sha256"]!=native.sha(config):raise ValueError("A/B source configuration differs")
    protocol=dict(version=1,episodes=list(range(48)),plan_call=0,full_horizon=30,prefix_limit=15,
        cem_rounds=15,samples_per_round=300,elites=10,captured_occurrences_per_method=4515,
        first_population_control="All 300 initial samples match exactly; native outcomes checked on every shared route",
        compared_search="Original A learned-dynamics-guided CEM versus independent B oracle-dynamics-guided CEM at identical initial states",
        common_native_criterion="Any success within horizon; earliest first hit among successes; smaller terminal TCP distance among failures",
        native_threshold_m=.05,a_args_sha256=native.sha(config),b_protocol_sha256=native.sha(b_protocol_path),
        compute_matching="Equal proposal budget; oracle simulation and learned prediction do not have matched wall-clock costs",
        new_search=False,new_encoder_or_model_run=False)
    database=output/"scores.sqlite3"
    with native.connect(database) as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.executescript("CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT); CREATE TABLE IF NOT EXISTS pools(episode INTEGER PRIMARY KEY,a_sha TEXT,b_sha TEXT,total INTEGER,complete INTEGER,source TEXT,control TEXT); CREATE TABLE IF NOT EXISTS scores(episode INTEGER,route_id TEXT,payload TEXT,PRIMARY KEY(episode,route_id));")
        encoded=json.dumps(protocol,sort_keys=True)
        old=db.execute("SELECT value FROM metadata WHERE key='protocol'").fetchone()
        if old and old[0]!=encoded:raise ValueError("Changed comparison protocol; use another output directory")
        db.execute("INSERT OR IGNORE INTO metadata VALUES ('protocol',?)",(encoded,))
    native.write_json(output/"protocol.json",protocol)
    lock=output/"active_process.json"
    if lock.exists():
        old=json.loads(lock.read_text())
        if psutil.pid_exists(old["pid"]) and psutil.Process(old["pid"]).create_time()==old["created"]:
            raise RuntimeError("The initial-search analysis already has a running process")
    lock.write_text(json.dumps(dict(pid=os.getpid(),created=psutil.Process().create_time())))
    awake=bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)) if os.name=="nt" else False
    episodes=list(range(48))[:args.max_pools] if args.max_pools is not None else list(range(48))
    try:
        with ProcessPoolExecutor(max_workers=args.workers,initializer=native.initialize,initargs=(str(config),)) as executor:
            pending=[executor.submit(score_episode,str(runs),str(database),ep) for ep in episodes]
            for future in as_completed(pending):
                result=future.result()
                with native.connect(database) as db:
                    completed=db.execute("SELECT count(*) FROM pools WHERE complete=1").fetchone()[0]
                    count=db.execute("SELECT count(*) FROM scores").fetchone()[0]
                status=dict(pid=os.getpid(),completed_episodes=completed,expected_episodes=48,complete=completed==48,
                    scored_unique_b_routes=count,last_completed=result)
                native.write_json(output/"status.json",status)
                print(json.dumps(status),flush=True)
    finally:
        if awake:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        lock.unlink(missing_ok=True)


if __name__=="__main__":main()

"""Replay B's saved candidate prefixes after a successful continuation was lost.

No new CEM search, encoder evaluation, or controller intervention. Each worker
uses the original native MetaWorldWrapper/TimeLimit, without pixel rendering or
tensor conversion. Native trajectories must match saved reference traces first.
Resume-safe SQLite commits retain per-route outcomes in this analysis directory.
"""

import argparse
import atexit
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import ctypes
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import sqlite3
from time import perf_counter

import numpy as np
from omegaconf import OmegaConf
import psutil
import torch


PROJECT = Path(__file__).resolve().parents[4]
DEFAULT_RUNS = PROJECT / "runs/reach_wall/simu_env_planning"
ENV = None


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle,"sha256").hexdigest()


def write_json(path, data):
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(data,indent=2)+"\n",encoding="utf-8")
    os.replace(temporary,path)


def targets(runs):
    path=runs.parent/"analysis/prefix_vs_full/future_only_successes.csv"
    with path.open(newline="") as handle:
        selected=[r for r in csv.DictReader(handle) if r["method"]=="B"]
    result=sorted({(int(r["episode"]),int(r["plan_call"])+1) for r in selected})
    if len(result)!=47 or len(selected)!=47:
        raise ValueError("Expected the 47 documented B successor decision states")
    return result,path


def connect(path):
    connection=sqlite3.connect(path,timeout=60)
    connection.execute("PRAGMA busy_timeout=60000")
    return connection


def initialize(config):
    global ENV
    torch.set_num_threads(1)
    logging.basicConfig(level=logging.ERROR)
    from evals.simu_env_planning.envs.metaworld import make_env
    cfg=OmegaConf.load(config)
    cfg.task_specification.done_at_succ=False
    ENV=make_env(cfg)
    ENV.reset(seed=1)
    atexit.register(ENV.close)


def restore(decision):
    observed=ENV.set_replay_state(decision["simulator_snapshot"])
    np.testing.assert_allclose(observed,decision["initial_state"].numpy(),atol=1e-7,rtol=0)
    ENV.set_elapsed_steps(decision["episode_steps_elapsed"])
    random.setstate(decision["rng"]["python"])
    np.random.set_state(decision["rng"]["numpy"])
    torch.set_rng_state(decision["rng"]["torch_cpu"])


def rollout(decision, actions, full_required=False):
    raw=np.asarray(actions,dtype=np.float32)
    if raw.ndim!=2 or raw.shape[1]!=4 or len(raw) not in (15,30):
        raise ValueError("Expected a 15-action tail or 30-action new candidate")
    if not np.isfinite(raw).all() or (np.abs(raw)>1.000001).any():
        raise ValueError("Saved candidate violates native action bounds")
    restore(decision)
    first=None
    trace=[]
    for step,action in enumerate(raw,1):
        _,_,done,truncated,info=ENV.step(action)
        distance,success=float(info["obj_to_target"]),bool(info["success"])
        if not np.isfinite(distance) or distance<0 or success!=(distance<=.05):
            raise ValueError("Invalid native measurement")
        trace.append(distance)
        if success and first is None:
            first=step
        if step==15 and len(raw)==30 and first is None and not full_required:
            break
        if (done or truncated) and step<len(raw):
            raise ValueError("Unexpected native truncation")
    prefix_hit=first if first is not None and first<=15 else None
    full_known=len(trace)==len(raw)
    return dict(prefix_success=prefix_hit is not None,prefix_first_hit=prefix_hit,
        prefix_endpoint_distance_m=trace[14],
        prefix_stopped_distance_m=trace[prefix_hit-1] if prefix_hit else trace[14],
        full_outcome_available=full_known,
        full_first_hit=first if full_known else None,
        full_success=first is not None if full_known else None,
        full_endpoint_distance_m=trace[-1] if full_known else None,
        full_endpoint_success=(trace[-1]<=.05) if full_known else None,
        simulated_actions=len(trace),distance_trace_m=trace)


def check_trace(outcome, reference):
    distances=[float(r["distance"]) for r in reference]
    np.testing.assert_allclose(outcome["distance_trace_m"],distances,atol=1e-8,rtol=0)
    if [d<=.05 for d in outcome["distance_trace_m"]]!=[bool(r["success"]) for r in reference]:
        raise ValueError("Replayed native success flags differ")
    return float(np.max(np.abs(np.asarray(outcome["distance_trace_m"])-distances)))


def route_id(actions):
    return hashlib.sha256(np.ascontiguousarray(actions,dtype=np.float32).tobytes()).hexdigest()


def score_pool(runs_string, database_string, episode, call):
    runs,db_path=Path(runs_string),Path(database_string)
    path=runs/f"closed_loop_b/ep_{episode}/plan_{call:03d}.pt"
    previous_path=path.with_name(f"plan_{call-1:03d}.pt")
    source_sha,previous_sha=sha(path),sha(previous_path)
    db=connect(db_path)
    existing=db.execute("SELECT source_sha,previous_sha,complete,control FROM pools WHERE episode=? AND plan_call=?",(episode,call)).fetchone()
    if existing and existing[:2]!=(source_sha,previous_sha):
        raise ValueError("Saved B decisions changed; use a new analysis output")
    if existing and existing[2]:
        db.close()
        return dict(episode=episode,plan_call=call,cached=True,**json.loads(existing[3])["pool_counts"])
    saved=torch.load(path,map_location="cpu",weights_only=False)
    previous=torch.load(previous_path,map_location="cpu",weights_only=False)
    decision=saved["decision"]
    if saved["branch"]!="B_closed_loop" or saved["episode"]!=episode or saved["plan_call"]!=call:
        raise ValueError("Wrong B decision bundle")
    if decision["episode_steps_elapsed"] not in (15,30,45):
        raise ValueError("Unexpected successor-state action index")
    candidates=saved["candidates"]
    if len(candidates)!=4515:
        raise ValueError("Incomplete B candidate capture")
    selected=[c for c in candidates if c["is_deployed_output"]]
    old_mean=[c for c in previous["candidates"] if c["is_deployed_output"]]
    if len(selected)!=1 or len(old_mean)!=1:
        raise ValueError("Missing selected final means")
    old_metrics=previous["final_mean_rollout"]["metrics"]
    if not old_metrics["task_success"] or old_metrics["prefix_success"] or old_metrics["prefix_steps"]!=15:
        raise ValueError("Previous selected plan does not supply the known successful continuation")
    old_raw=old_mean[0]["actions_simulator"].numpy()
    tail=old_raw[15:]
    if len(tail)!=15:
        raise ValueError("Expected a 15-action remaining successful continuation")
    previous_check=rollout(previous["decision"],old_raw,True)
    e0=check_trace(previous_check,previous["final_mean_rollout"]["trace"])
    tail_check=rollout(decision,tail,True)
    e1=check_trace(tail_check,previous["final_mean_rollout"]["trace"][15:])
    selected_check=rollout(decision,selected[0]["actions_simulator"].numpy(),True)
    e2=check_trace(selected_check,saved["final_mean_rollout"]["trace"])
    if not tail_check["prefix_success"] or selected_check["prefix_success"]:
        raise ValueError("Known-tail feasibility or actual selected-prefix failure did not reproduce")
    with (path.parent/"execution.csv").open(newline="") as handle:
        actual=[r for r in csv.DictReader(handle) if int(r["plan_call"])==call]
    np.testing.assert_allclose(selected_check["distance_trace_m"][:15],
        [float(r["tcp_goal_distance_m"]) for r in actual],atol=1e-8,rtol=0)
    routes={}
    best_by_round={}
    for candidate in candidates:
        if candidate["origin"]!="updated_mean":
            round_number=candidate["cem_round"]
            if round_number not in best_by_round or candidate["b_cost"]<best_by_round[round_number]["b_cost"]:
                best_by_round[round_number]=candidate
    best_names={c["name"] for c in best_by_round.values()}
    for candidate in candidates:
        raw=candidate["actions_simulator"].numpy()
        key=route_id(raw)
        need_full=bool(candidate["is_elite"] or candidate["origin"]=="updated_mean" or candidate["name"] in best_names)
        if key in routes:
            routes[key]["full_required"] |= need_full
        else:
            routes[key]=dict(actions=raw,full_required=need_full)
    source=dict(episode=episode,plan_call=call,start_action=decision["episode_steps_elapsed"],
        source=str(path),previous_source=str(previous_path),source_sha256=source_sha,previous_sha256=previous_sha,
        known_tail_first_hit=tail_check["prefix_first_hit"],known_tail_prefix_hash=route_id(tail),
        native_reference_max_error_m=max(e0,e1,e2),selected_mean_prefix_success=False,
        original_candidate_occurrences=len(candidates),unique_full_routes=len(routes),
        full_outcome_policy="Full native continuation for successful prefixes, all elites, all updated means, and each round's lowest-cost sample. Others stop at prefix action 15.")
    with db:
        db.execute("INSERT OR IGNORE INTO pools VALUES (?,?,?,?,?,?,?,?)",
            (episode,call,source_sha,previous_sha,len(routes),0,json.dumps(source),None))
    completed={key:json.loads(payload) for key,payload in db.execute("SELECT route_id,payload FROM scores WHERE episode=? AND plan_call=?",(episode,call))}
    pending=[]
    started=perf_counter()
    for key,route in routes.items():
        if key in completed:
            continue
        outcome=rollout(decision,route["actions"],route["full_required"])
        completed[key]=outcome
        pending.append((episode,call,key,json.dumps(outcome)))
        if len(pending)>=100:
            with db:db.executemany("INSERT INTO scores VALUES (?,?,?,?)",pending)
            pending=[]
    if pending:
        with db:db.executemany("INSERT INTO scores VALUES (?,?,?,?)",pending)
    if len(completed)!=len(routes):
        raise ValueError("Missing native prefix scores")
    counts=dict(unique_routes=len(routes),unique_routes_with_successful_prefix=sum(r["prefix_success"] for r in completed.values()),
        candidate_occurrences=len(candidates),successful_prefix_occurrences=sum(completed[route_id(c["actions_simulator"].numpy())]["prefix_success"] for c in candidates),
        native_steps=sum(r["simulated_actions"] for r in completed.values()))
    control=dict(source, pool_counts=counts,worker_seconds=perf_counter()-started)
    with db:db.execute("UPDATE pools SET complete=1,control=? WHERE episode=? AND plan_call=?",(json.dumps(control),episode,call))
    db.close()
    return dict(episode=episode,plan_call=call,cached=False,**counts)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir",type=Path,default=DEFAULT_RUNS)
    parser.add_argument("--output-dir",type=Path)
    parser.add_argument("--workers",type=int,default=3,choices=range(1,5))
    parser.add_argument("--max-pools",type=int)
    args=parser.parse_args()
    runs=args.runs_dir.resolve()
    output=(args.output_dir or runs.parent/"analysis/b_prefix_candidates").resolve()
    output.mkdir(parents=True,exist_ok=True)
    cases,source=targets(runs)
    config=runs/"bounded_a_i15_n300_ep48/args_eval.yaml"
    expected=dict(schema_version=1,source_success_cases=str(source),source_success_cases_sha256=sha(source),
        config_sha256=sha(config),target_pools=cases,prefix_actions=15,full_horizon_actions=30,
        criterion="Native TCP distance <=0.05m at any prefix action",new_search=False,encoder_loaded=False,
        native_replay="Original MetaWorldWrapper + TimeLimit, no pixel rendering or TensorWrapper; reference trajectories checked before each pool",
        observational_scope="B's next planning call after its chosen full plan succeeds beyond the executed prefix; outcome-conditioned diagnostic, not representative of every B state")
    database=output/"scores.sqlite3"
    with connect(database) as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.executescript("CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT); CREATE TABLE IF NOT EXISTS pools(episode INTEGER,plan_call INTEGER,source_sha TEXT,previous_sha TEXT,total INTEGER,complete INTEGER,source TEXT,control TEXT,PRIMARY KEY(episode,plan_call)); CREATE TABLE IF NOT EXISTS scores(episode INTEGER,plan_call INTEGER,route_id TEXT,payload TEXT,PRIMARY KEY(episode,plan_call,route_id));")
        old=db.execute("SELECT value FROM metadata WHERE key='protocol'").fetchone()
        encoded=json.dumps(expected,sort_keys=True)
        if old and old[0]!=encoded:
            raise ValueError("Analysis source/protocol differs; use a new output directory")
        db.execute("INSERT OR IGNORE INTO metadata VALUES ('protocol',?)",(encoded,))
    write_json(output/"protocol.json",expected)
    lock=output/"active_process.json"
    if lock.exists():
        old=json.loads(lock.read_text())
        if psutil.pid_exists(old["pid"]) and psutil.Process(old["pid"]).create_time()==old["created"]:
            raise RuntimeError("This analysis already has a running process")
    lock.write_text(json.dumps(dict(pid=os.getpid(),created=psutil.Process().create_time())))
    awake=bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)) if os.name=="nt" else False
    chosen=cases[:args.max_pools] if args.max_pools is not None else cases
    try:
        with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize,initargs=(str(config),)) as executor:
            pending=[executor.submit(score_pool,str(runs),str(database),ep,call) for ep,call in chosen]
            for future in as_completed(pending):
                result=future.result()
                with connect(database) as db:
                    completed=db.execute("SELECT count(*) FROM pools WHERE complete=1").fetchone()[0]
                    scored=db.execute("SELECT count(*) FROM scores").fetchone()[0]
                status=dict(pid=os.getpid(),completed_pools=completed,expected_pools=len(cases),complete=completed==len(cases),
                    scored_unique_routes=scored,last_completed=result)
                write_json(output/"status.json",status)
                print(json.dumps(status),flush=True)
    finally:
        if awake:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        lock.unlink(missing_ok=True)


if __name__=="__main__":main()

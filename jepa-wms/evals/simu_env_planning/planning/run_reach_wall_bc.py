"""Resume B, then run complete C analysis automatically, with a status file."""

import argparse
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from time import sleep

import psutil

REPO = Path(__file__).resolve().parents[3]
DEFAULT_RUN = REPO.parent / "runs/reach_wall/simu_env_planning/bounded_a_i15_n300_ep48"


def save_status(path, status):
    status["updated_utc"] = datetime.now(timezone.utc).isoformat()
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(status,indent=2),encoding="utf-8")
    os.replace(temporary,path)


def b_progress(run):
    path = run/"branch_b/scores.sqlite3"
    if not path.exists():
        return {}
    connection = sqlite3.connect("file:"+path.as_posix()+"?mode=ro",uri=True)
    try:
        rows = connection.execute("SELECT p.episode,p.plan_call,p.total,count(s.candidate_index) FROM pools p LEFT JOIN scores s ON p.episode=s.episode AND p.plan_call=s.plan_call GROUP BY p.episode,p.plan_call").fetchall()
        return dict(b_complete_pools=sum(total==count for _,_,total,count in rows),
                    b_scored_occurrences=sum(count for _,_,_,count in rows))
    finally:
        connection.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path,default=DEFAULT_RUN)
    args = parser.parse_args(argv)
    run = args.run_dir.resolve()
    status_path = run/"bc_pipeline_status.json"
    if status_path.exists():
        previous = json.loads(status_path.read_text())
        try:
            old = psutil.Process(previous["pid"])
            if old.create_time()==previous.get("process_created") and previous.get("phase") in ("B","C"):
                raise RuntimeError("This B/C pipeline is already running")
        except (psutil.NoSuchProcess,KeyError):
            pass
    status = dict(pid=os.getpid(),process_created=psutil.Process().create_time(),phase="starting",
                  started_utc=datetime.now(timezone.utc).isoformat(),run_dir=str(run))
    awake = bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)) if os.name=="nt" else False
    status["automatic_sleep_prevention_requested"] = awake
    child = None
    log_path = run/"bc_pipeline.log"
    try:
        with log_path.open("a",encoding="utf-8") as handle:
            for phase,module,extra in (
                ("B","evals.simu_env_planning.planning.reach_wall_b",["--all"]),
                ("C","evals.simu_env_planning.planning.reach_wall_c",[]),
            ):
                status["phase"] = phase
                command = [sys.executable,"-u","-X","utf8","-m",module,"--run-dir",str(run),*extra]
                print(f"Starting {phase}; detailed log: {log_path}",flush=True)
                handle.write("\n"+datetime.now(timezone.utc).isoformat()+" START "+phase+"\n");handle.flush()
                child = subprocess.Popen(command,cwd=REPO,stdout=handle,stderr=subprocess.STDOUT,
                                         creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
                status["child_pid"] = child.pid
                while child.poll() is None:
                    status.update(b_progress(run))
                    save_status(status_path,status)
                    sleep(5)
                if child.returncode:
                    raise RuntimeError(f"{phase} exited with code {child.returncode}; see {log_path}")
            summary = json.loads((run/"branch_c/summary.json").read_text())
            if not summary["complete"]:
                raise RuntimeError("C did not produce a complete analysis")
            status.update(b_progress(run))
            status.update(phase="complete",results=str(run/"branch_c/results.md"),c_summary=summary)
            print("B and C complete. Results:",status["results"],flush=True)
    except BaseException as exc:
        if child is not None and child.poll() is None:
            child.terminate()
        status.update(phase="failed",error=str(exc))
        raise
    finally:
        save_status(status_path,status)
        if awake:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=="__main__":
    main()

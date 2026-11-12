"""Independent checks of the published data, native traces and headline results."""
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]


def rows(relative):
    with (ROOT / relative).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_json(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def close(actual, expected):
    if not math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=1e-10):
        raise AssertionError((actual, expected))


def check_hashes_and_links():
    manifest = read_json("data/manifest.json")
    for entry in manifest["copied_files"]:
        path = ROOT / entry["published"]
        assert path.is_file(), path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["published_sha256"], path
    links = 0
    for path in [ROOT / "README.md", ROOT / "THIRD-PARTY-LICENSES.md", *sorted((ROOT / "docs").glob("*.md")), ROOT / "data/README.md", ROOT / "notes/README.md"]:
        for target in re.findall(r"!?\[[^\]]*\]\(([^\s)]+)\)", path.read_text(encoding="utf-8")):
            if "://" in target or target.startswith(("#", "mailto:")):
                continue
            assert (path.parent / target.split("#")[0]).exists(), (path, target)
            links += 1
    for path in ROOT.rglob("*.py"):
        if ".git" not in path.parts and ".venv" not in path.parts:
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    return dict(copied_files=len(manifest["copied_files"]), local_document_links=links)


def check_wall():
    source = rows("data/wall/scored_candidates.csv")
    assert len(source) == 6500
    unique = {}
    for row in source:
        unique.setdefault((int(row["episode"]), row["route_id"]), row)
    assert len(unique) == 6410
    summary = read_json("data/wall/terminal/summary.json")
    selected = rows("data/wall/terminal/selections.csv")
    rhos = []
    for ep in range(10):
        pool = [r for (episode, _), r in unique.items() if episode == ep]
        assert len(pool) == 641
        best = min(float(r["end_distance"]) for r in pool)
        rhos.append(float(spearmanr([float(r["b_cost"]) for r in pool], [float(r["end_distance"]) for r in pool]).statistic))
        for choice in (r for r in selected if int(r["episode"]) == ep):
            native = unique[ep, choice["route_id"]]
            close(choice["end_distance"], native["end_distance"])
            close(choice["terminal_distance_regret"], float(native["end_distance"]) - best)
    close(np.mean(rhos), summary["correlations"]["spearman_b_terminal"]["mean"])
    b = [r for r in selected if r["selection"] == "min_b"]
    concentration = sum(float(r["terminal_distance_regret"]) for r in b if int(r["episode"]) in (3, 4)) / sum(float(r["terminal_distance_regret"]) for r in b)
    assert round(concentration * 100, 2) == 98.64
    geometry = {int(r["episode"]): r for r in rows("data/wall/routes/episode_geometry.csv")}
    with np.load(ROOT / "data/wall/routes/trajectories.npz", allow_pickle=False) as cache:
        assert cache["states"].shape == (6410, 31, 2)
        assert len(set(zip(cache["episode"].tolist(), cache["route_id"].tolist()))) == 6410
        for ep, route_id, states in zip(cache["episode"], cache["route_id"], cache["states"]):
            g = geometry[int(ep)]
            goal = np.array([float(g["goal_x"]), float(g["goal_y"])])
            distance = np.linalg.norm(states[-1].astype(float) - goal)
            np.testing.assert_allclose(distance, float(unique[int(ep), route_id]["end_distance"]), atol=5e-5, rtol=2e-6)
    return dict(occurrences=len(source), unique_routes=len(unique), b_regret_ep3_ep4_pct=concentration * 100)


def check_native_execution():
    data = rows("data/reach_wall/closed_loop/distance_traces.csv")
    published = read_json("data/reach_wall/closed_loop/summary.json")
    result = {}
    for method in ("A", "B", "C"):
        final, successes, actions = [], 0, 0
        for ep in range(48):
            native = rows(f"data/reach_wall/executions/{method}/ep_{ep}.csv")
            trace = [r for r in data if r["method"] == method and int(r["episode"]) == ep]
            assert len(native) == len(trace) <= 100
            flags = []
            for index, (raw, record) in enumerate(zip(native, trace), 1):
                assert int(raw["simulator_step"]) == int(record["raw_action"]) == index
                distance = float(raw["tcp_goal_distance_m"])
                close(record["tcp_goal_distance_m"], distance)
                assert raw["success"] == record["native_success"]
                success = raw["success"] == "True"
                assert success == (distance <= .05)
                flags.append(success)
            assert not any(flags[:-1])
            assert flags[-1] or len(native) == 100
            successes += flags[-1]
            actions += len(native)
            final.append(float(native[-1]["tcp_goal_distance_m"]) * 100)
        assert successes == published[method]["successes"]
        assert actions == published[method]["executed_actions"]
        close(np.mean(final), published[method]["mean_final_distance_cm"])
        result[method] = dict(successes=successes, actions=actions, mean_final_distance_cm=float(np.mean(final)))
    assert len(data) == 11109
    assert [result[m]["successes"] for m in ("A", "B", "C")] == [6, 0, 48]
    return result


def check_shared_pool():
    prefix = "data/reach_wall/shared_pool/terminal"
    summary = read_json(f"{prefix}/summary.json")
    pools = rows(f"{prefix}/pool_summary.csv")
    selections = rows(f"{prefix}/selections.csv")
    assert len(pools) == 672 and len(selections) == 3360
    expected = {"full_horizon": [14, 13, 35, 243], "executed_prefix": [6, 6, 7, 103]}
    for scope in expected:
        batch = [r for r in pools if r["scope"] == scope]
        lookup = {(int(r["episode"]), int(r["plan_call"])): r for r in batch}
        assert len(lookup) == 336
        for label, successes in zip(("a_deployed_mean", "a_pool_min", "b_pool_min", "c_terminal_best"), expected[scope]):
            selected = [r for r in selections if r["scope"] == scope and r["selection"] == label]
            assert len(selected) == 336
            assert sum(float(r["terminal_distance_m"]) <= .05 for r in selected) == successes
            for row in selected:
                best = float(lookup[int(row["episode"]), int(row["plan_call"])]["best_terminal_distance_m"])
                close(row["terminal_regret_m"], float(row["terminal_distance_m"]) - best)
            close(np.mean([float(r["terminal_regret_m"]) for r in selected]), summary["scopes"][scope]["selections"][label]["terminal_regret_m"]["mean"])
        for name, values in summary["scopes"][scope]["correlations"].items():
            close(np.mean([float(r["spearman_" + name]) for r in batch]), values["mean"])
    return dict(decision_pools=336, terminal_success_counts=expected)


def check_audits():
    audit = read_json("data/reach_wall/b_prefix_candidates/summary.json")
    pools = rows("data/reach_wall/b_prefix_candidates/pool_summary.csv")
    rounds = rows("data/reach_wall/b_prefix_candidates/cem_round_aggregate.csv")
    assert len(pools) == 47
    assert len({int(r["episode"]) for r in pools}) == 43
    assert sum(int(r["successful_prefix_occurrences"]) > 0 for r in pools) == audit["pools_with_any_successful_prefix"] == 44
    matched_terminal = sum(r["successful_prefix_with_goal_terminal_exists"] == "True"
                           and r["returned_mean_full_endpoint_success"] == "False" for r in pools)
    assert matched_terminal == audit["pools_with_goal_terminal_successful_prefix_but_returned_terminal_outside"] == 41
    assert sum(int(r["successful_prefix_final_elites"]) for r in pools) == 0
    assert sum(r["returned_mean_prefix_success"] == "True" for r in pools) == 0
    assert all(1 <= int(r["known_old_tail_first_hit"]) <= 15 for r in pools)
    assert sum(int(r["successful_prefix_samples"]) for r in pools) == sum(int(r["successful_prefix_samples"]) for r in rounds) == 3303
    assert [int(rounds[i]["successful_prefix_samples"]) for i in (0, -1)] == [946, 7]
    initial = rows("data/reach_wall/initial_search_ab/pool_summary.csv")
    coverage = {method: sum(r["full_pool_contains_success"] == "True" for r in initial if r["method"] == method) for method in ("A", "B")}
    assert coverage == {"A": 0, "B": 4}
    selected = rows("data/reach_wall/prefix_vs_full/selected_plans.csv")
    assert len(selected) == 672
    return dict(b_audited_states=47, pools_with_successful_prefix=44,
                pools_with_matched_terminal_selection_gap=matched_terminal, initial_search_coverage=coverage)


def main():
    result = dict(files=check_hashes_and_links(), wall=check_wall(), native_execution=check_native_execution(),
                  shared_pool=check_shared_pool(), audits=check_audits(), passed=True)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()

"""Plot every saved Wall route across all ten episodes, highlighting B/C picks.

Uses cached exact trajectories when available; otherwise replays saved actions
in the original simulator. No planner, world model, or encoder is loaded.
Run in AI: python -m evals.simu_env_planning.planning.plot_wall_pool_routes
"""

import argparse
import hashlib
import html
import json
import logging
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
import numpy as np
from omegaconf import OmegaConf
import torch

from .analyze_candidates import deduplicate_routes, group_by, load_scores
from .common.parser import parse_cfg
from .inspect_wall_routes import DEFAULT_RUN, background, check_outcome, read_csv, replay, save


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cached_routes(path, protocol_path, source_hash, bundle_hashes):
    if not path.exists() or not protocol_path.exists():
        return {}
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["source_scores_sha256"] != source_hash or protocol["bundle_sha256"] != bundle_hashes:
        raise ValueError("Trajectory cache provenance does not match the current scores/bundles")
    if "trajectory_cache_sha256" in protocol and protocol["trajectory_cache_sha256"] != digest(path):
        raise ValueError("Trajectory cache hash differs from its recorded provenance")
    with np.load(path, allow_pickle=False) as data:
        keys = [(int(ep), route_id) for ep, route_id in zip(data["episode"], data["route_id"])]
        if len(set(keys)) != len(keys):
            raise ValueError("Repeated cached episode/route identity")
        return {key: torch.from_numpy(states.copy()) for key, states in zip(keys, data["states"])}


def panel(ax, ep, bundle, rows, paths, selected, native, compact):
    background(ax, bundle, native, label=not compact)
    ax.add_collection(LineCollection([paths[ep, row["route_id"]].numpy() for row in rows],
                                     colors="#666666", linewidths=.55, alpha=.045,
                                     zorder=2, rasterized=True))
    b = paths[ep, selected[ep, "min_b"]["route_id"]].numpy()
    c = paths[ep, selected[ep, "min_terminal"]["route_id"]].numpy()
    same = np.array_equal(b, c)
    # A dashed blue line over a solid green line keeps both selections visible
    # wherever their physical trajectories coincide.
    for xy, color, style, width in ((c, "#009E73", "-", 3.2), (b, "#0072B2", "--", 2.1)):
        line, = ax.plot(xy[:, 0], xy[:, 1], color=color, linestyle=style, linewidth=width, zorder=5)
        line.set_path_effects([pe.Stroke(linewidth=width + 1.2, foreground="white"), pe.Normal()])
    ax.scatter(*c[-1], c="#009E73", s=85 if compact else 135, marker="D", edgecolors="white", linewidths=.7, zorder=8)
    ax.scatter(*b[-1], c="#0072B2", s=48 if compact else 80, marker="X", edgecolors="white", linewidths=.6, zorder=9)
    ax.scatter(*bundle["initial_state"].numpy(), c="black", s=28 if compact else 45,
               edgecolors="white", linewidths=.7, zorder=10)
    b_distance, c_distance = (float(selected[ep, name]["end_distance"]) for name in ("min_b", "min_terminal"))
    if compact:
        title = f"Episode {ep} | {len(rows)} routes\nd(B)={b_distance:.2f}  d(C)={c_distance:.2f}"
    else:
        title = f"Episode {ep}: all {len(rows)} unique candidate routes\nB terminal distance={b_distance:.2f} | C terminal distance={c_distance:.2f}"
    ax.set_title(title, fontsize=10 if compact else 12, pad=10)
    if same:
        ax.text(.97, .97, "B = C trajectory", transform=ax.transAxes, ha="right", va="top",
                fontsize=8 if compact else 10, bbox=dict(facecolor="white", edgecolor=".8", alpha=.94, pad=3), zorder=11)


def legend(fig):
    handles = [Line2D([], [], color=".5", linewidth=1, label="All candidate routes"),
               Line2D([], [], color="#0072B2", linestyle="--", linewidth=2, marker="X", label="B: minimum terminal latent cost"),
               Line2D([], [], color="#009E73", linewidth=3, marker="D", label="C: minimum terminal physical distance"),
               Line2D([], [], color="black", marker="o", linestyle="None", label="Start"),
               Line2D([], [], color="#F6C344", marker="*", markeredgecolor="black", markersize=10, linestyle="None", label="Goal")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .035),
               ncol=3, frameon=False, fontsize=9)
    fig.text(.5, .012, "Same coordinate scale in every panel. Full 30-action diagnostic routes, including continuation after earlier success.\n"
             "Gray overlap is not probability. Matching B/C trajectories show both colors; duplicated action sequences are drawn once.",
             ha="center", fontsize=9, color=".3")


def plot_overview(pools, bundles, paths, selected, native, output):
    fig, axes = plt.subplots(2, 5, figsize=(18, 9.6), squeeze=False)
    for ax, ep in zip(axes.flat, sorted(pools)):
        panel(ax, ep, bundles[ep], pools[ep], paths, selected, native, compact=True)
    fig.suptitle("All candidate routes across ten Wall episodes | terminal B/C selections", fontsize=16)
    legend(fig)
    # Reserve explicit title/tick space between square map panels.
    fig.subplots_adjust(left=.035, right=.99, top=.88, bottom=.16, hspace=.36, wspace=.22)
    save(fig, output, "all_candidate_routes_all_episodes")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    run = args.run_dir
    output = args.output_dir or run / "analysis_terminal/route_inspection"
    output.mkdir(parents=True, exist_ok=True)
    source = run / "scored_candidates.csv"
    source_hash = digest(source)
    rows = deduplicate_routes(load_scores(source))
    pools = group_by(rows, "episode")
    if sorted(pools) != list(range(10)):
        raise ValueError("This comparison expects the ten saved Wall episodes 0..9")
    selected_path = run / "analysis_terminal/selections.csv"
    selected = {(int(row["episode"]), row["selection"]): row for row in read_csv(selected_path)}
    bundle_paths = {ep: run / f"wall-base/ep_{ep}/branch_a/plan_000.pt" for ep in pools}
    bundle_hashes = {str(path.relative_to(run)): digest(path) for path in bundle_paths.values()}
    cache_path = output / "all_candidate_trajectories_all_episodes.npz"
    protocol_path = output / "all_candidate_routes_protocol.json"
    paths = cached_routes(cache_path, protocol_path, source_hash, bundle_hashes)
    if not paths:
        paths = cached_routes(output / "all_candidate_trajectories.npz", output / "protocol.json", source_hash, bundle_hashes)
    original_keys = set(paths)
    bundles = {}
    max_error = 0.
    torch.set_num_threads(1)
    logging.disable(logging.INFO)
    warnings.filterwarnings("ignore", message="To copy construct from a tensor")
    from evals.simu_env_planning.envs.init import make_env
    cfg = OmegaConf.load(run / "args_eval.yaml")
    cfg.work_dir = str(run)
    cfg.frameskip = cfg.model_kwargs.data.custom.frameskip
    env = make_env(parse_cfg(cfg))
    try:
        for ep, batch in sorted(pools.items()):
            bundle = torch.load(bundle_paths[ep], map_location="cpu", weights_only=False)
            bundles[ep] = bundle
            candidates = {candidate["name"]: candidate for candidate in bundle["candidates"]}
            reused = 0
            for row in batch:
                key = ep, row["route_id"]
                candidate = candidates[row["candidate"]]
                if hashlib.sha256(candidate["actions_normalized"].contiguous().numpy().tobytes()).hexdigest() != row["route_id"]:
                    raise ValueError("Saved action bytes do not match the scored route identity")
                if key in paths:
                    reused += 1
                else:
                    paths[key], _ = replay(env, bundle, candidate)
                states = paths[key]
                if states.shape != (31, 2):
                    raise ValueError("Expected start plus all 30 raw action outcomes")
                torch.testing.assert_close(states[0], bundle["initial_state"], rtol=0, atol=0)
                max_error = max(max_error, check_outcome(states, row, bundle))
            for name, field in (("min_b", "b_cost"), ("min_terminal", "end_distance")):
                choice = selected[ep, name]
                if float(choice[field]) != min(row[field] for row in batch):
                    raise ValueError("Highlighted selection is not a pooled minimum")
                if (ep, choice["route_id"]) not in paths:
                    raise ValueError("Highlighted route missing from the replayed pool")
            print(f"Episode {ep}: {len(batch)} complete routes verified ({reused} cached, {len(batch)-reused} replayed).", flush=True)
        if len(paths) != len(rows):
            raise ValueError("Trajectory cache contains missing or extraneous routes")
        np.savez_compressed(cache_path,
            episode=np.array([row["episode"] for row in rows]),
            route_id=np.array([row["route_id"] for row in rows]),
            candidate=np.array([row["candidate"] for row in rows]),
            states=np.stack([paths[row["episode"], row["route_id"]].numpy() for row in rows]),
            b_cost=np.array([row["b_cost"] for row in rows]),
            end_distance=np.array([row["end_distance"] for row in rows]))
        print("Rendering the combined comparison and ten individual figures...", flush=True)
        with plt.rc_context({"font.size": 10, "pdf.fonttype": 42}):
            plot_overview(pools, bundles, paths, selected, env.proprio_env.unwrapped, output)
            for ep in sorted(pools):
                fig, ax = plt.subplots(figsize=(8, 8.5))
                panel(ax, ep, bundles[ep], pools[ep], paths, selected, env.proprio_env.unwrapped, compact=False)
                legend(fig)
                fig.tight_layout(rect=(0, .15, 1, .98))
                save(fig, output, f"all_candidate_routes_ep_{ep}")
                print(f"Saved individual episode {ep}.", flush=True)
    finally:
        env.close()
    if digest(source) != source_hash or any(digest(path) != bundle_hashes[str(path.relative_to(run))] for path in bundle_paths.values()):
        raise ValueError("Input scores or action bundles changed during plotting")
    protocol = dict(source_scores_sha256=source_hash, selections_sha256=digest(selected_path),
                    bundle_sha256=bundle_hashes, trajectory_cache_sha256=digest(cache_path),
                    episodes=sorted(pools), unique_routes=len(rows), state_shape=[len(rows), 31, 2],
                    cached_routes_reused=len(original_keys), new_routes_replayed=len(paths)-len(original_keys),
                    maximum_saved_distance_error=max_error,
                    b_selection="minimum saved full-horizon actual-endpoint latent cost",
                    c_selection="minimum saved full-horizon Euclidean goal distance; not first-hit C",
                    gray_routes="all unique captured action sequences, from all CEM rounds, once per episode",
                    replay="original native DotWall.step with saved raw actions, start, wall, and RNG; no clipping",
                    rerun_world_model=False, rerun_encoder=False)
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    gallery_path = output / "index.html"
    gallery = gallery_path.read_text(encoding="utf-8") if gallery_path.exists() else "<!doctype html><meta charset='utf-8'><title>Wall candidate routes</title>"
    marker = "<!-- all-episode candidate routes -->"
    if marker in gallery:
        gallery = gallery.split(marker)[0]
    gallery += marker + "<h1>All ten candidate pools</h1><p>Gray: all routes; dashed blue: terminal B; solid green: terminal C. Click any image to enlarge.</p>"
    for name in ["all_candidate_routes_all_episodes"] + [f"all_candidate_routes_ep_{ep}" for ep in sorted(pools)]:
        gallery += f"<section><h2>{html.escape(name.replace('_',' ').capitalize())}</h2><a href='{name}.png'><img style='width:100%' src='{name}.png'></a></section>"
    gallery_path.write_text(gallery, encoding="utf-8")
    print(f"All {len(rows):,} route trajectories verified. Maximum saved distance error: {max_error:g}.", flush=True)
    print(f"Combined figure: {output / 'all_candidate_routes_all_episodes.png'}", flush=True)


if __name__ == "__main__":
    main()

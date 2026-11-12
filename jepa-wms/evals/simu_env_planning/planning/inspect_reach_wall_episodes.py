"""Inspect paired actual Reach-Wall executions using saved logs and videos only.

Run in AI from jepa-wms:
    python -m evals.simu_env_planning.planning.inspect_reach_wall_episodes

The offline HTML viewer synchronizes videos by executed raw action, not by
planning call or wall-clock time. No simulator replay or model inference.
"""

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .plot_reach_wall_distance import DEFAULT_RUNS, METHODS, digest, load_traces, write_csv


REASONS = {
    0: "B endpoint near the median of B's 48 final distances",
    14: "A succeeds and B fails from the same initial state",
    18: "Largest B final distance among the 48 trials",
}
SAMPLE_STEPS = (0, 30, 60, 90)


def execution_records(runs, method, ep):
    folder = runs / METHODS[method]["folder"] / f"ep_{ep}"
    with (folder / "execution.csv").open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    starts = sorted({int(row["simulator_step"]) - 1 for index, row in enumerate(rows)
                     if index == 0 or row["plan_call"] != rows[index-1]["plan_call"]})
    if method == "A":
        videos = list((folder / "visualisation").glob("video_agent_goal_*.mp4"))
        if len(videos) != 1:
            raise ValueError(f"Expected one actual A video in {folder}")
        video = videos[0]
    else:
        video = folder / "execution.mp4"
    return rows, starts, video


def copy_video_and_frames(source, destination, actions):
    """Frame zero is initial observation; frame k follows actual raw action k."""
    with imageio.get_reader(source, format="FFMPEG") as reader:
        metadata = reader.get_meta_data()
        count = reader.count_frames()
        if count != actions + 1 or not math.isclose(metadata["fps"], 30):
            raise ValueError(f"Video/action alignment is unexpected: {source}")
        indices = sorted({0, actions, *(step for step in SAMPLE_STEPS if step <= actions)})
        frames = {step: reader.get_data(step) for step in indices}
        if any(frame.shape != (224, 224, 3) for frame in frames.values()):
            raise ValueError(f"Unexpected image shape: {source}")
    shutil.copyfile(source, destination)
    sha = digest(source)
    if digest(destination) != sha:
        raise AssertionError("Copied video changed")
    return frames, dict(source=str(source), sha256=sha, copied_to=str(destination),
                        frames=count, fps=metadata["fps"], frame_action_mapping="frame k follows raw action k; frame 0 is initial")


def distance_axes(axes, records, starts, full_limit):
    """Overlay the three controllers at this paired initial condition."""
    for ax, closeup in zip(axes, (False, True)):
        for method, trace in records.items():
            color = METHODS[method]["color"]
            cm = trace["distance_m"] * 100
            line, = ax.plot(trace["steps"], cm, color=color, lw=1.8, label=method)
            ax.scatter(trace["steps"][-1], cm[-1], s=35, edgecolor=color,
                       facecolor=color if trace["success"][-1] else "white", zorder=5)
            np.testing.assert_array_equal(line.get_xdata(), trace["steps"])
            np.testing.assert_array_equal(line.get_ydata(), cm)
        for step in sorted({step for values in starts.values() for step in values if step > 0}):
            ax.axvline(step, color=".8", ls=":", lw=.8, zorder=0)
        ax.axhspan(0, 5, color="#E8F3E9", alpha=.6, zorder=0)
        ax.axhline(5, color="#A12A32", ls="--", lw=1.2)
        ax.set(xlim=(0, 101), ylim=(3, 16) if closeup else (0, full_limit),
               xlabel="Executed raw simulator actions", ylabel="Native TCP-to-goal distance (cm)",
               title="Near-goal detail" if closeup else "Full distance range")
        ax.grid(axis="y", color=".9", lw=.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="upper right", ncol=3, frameon=False, fontsize=9)


def figures(ep, records, starts, frames, output):
    upper = 5 * math.ceil(max(float(t["distance_m"].max()*100) for t in records.values()) / 5)
    with plt.rc_context({"font.size":10, "pdf.fonttype":42}):
        curve, axes = plt.subplots(1, 2, figsize=(12, 4.3))
        distance_axes(axes, records, starts, upper)
        curve.suptitle(f"Episode {ep}: paired actual executions", fontsize=15)
        curve.text(.5, .02, "Dashed red: 5 cm success threshold. Dotted gray: replanning starts (only while that controller is active).\n"
                   "Curves end at actual stopping actions; filled endpoint = success, hollow endpoint = action budget.",
                   ha="center", fontsize=8, color=".3")
        curve.subplots_adjust(left=.065, right=.985, top=.80, bottom=.26, wspace=.23)
        for suffix in ("png", "pdf"):
            curve.savefig(output / f"episode_{ep}_curves.{suffix}", dpi=180, bbox_inches="tight")
        plt.close(curve)

        fig = plt.figure(figsize=(15, 12.3))
        layout = fig.add_gridspec(4, 1, height_ratios=[1.16, 1, 1, 1],
                                  left=.045, right=.99, top=.92, bottom=.055, hspace=.4)
        top = layout[0].subgridspec(1, 2, wspace=.18)
        axes = [fig.add_subplot(top[0, index]) for index in range(2)]
        distance_axes(axes, records, starts, upper)
        manifest = []
        for row, (method, trace) in enumerate(records.items(), 1):
            cells = layout[row].subgridspec(1, 5, wspace=.045)
            stop = int(trace["steps"][-1])
            for column, requested in enumerate((*SAMPLE_STEPS, None)):
                ax = fig.add_subplot(cells[0, column])
                step = stop if requested is None else requested
                ax.set_xticks([])
                ax.set_yticks([])
                ax.spines[:].set_visible(False)
                if column == 0:
                    ax.text(-.06, .5, method, color=METHODS[method]["color"], transform=ax.transAxes,
                            ha="right", va="center", fontsize=17, fontweight="bold")
                if step > stop:
                    ax.set_facecolor("#F3F4F5")
                    ax.text(.5, .5, f"Stopped at action {stop}\nNo later execution", transform=ax.transAxes,
                            ha="center", va="center", color=".4", fontsize=10)
                    ax.set_title(f"Action {step}", fontsize=10)
                    continue
                ax.imshow(frames[method][step])
                if requested is None:
                    ending = "success" if trace["success"][-1] else "budget"
                    title = f"Actual stop: {step} ({ending})"
                else:
                    title = "Initial observation" if step == 0 else f"Action {step}"
                if step:
                    title += f"\n{trace['distance_m'][step-1]*100:.2f} cm"
                else:
                    title += "\nStep-0 distance not logged"
                ax.set_title(title, fontsize=9)
                manifest.append(dict(episode=ep, method=method, requested_step=requested,
                                     video_frame_index=step, logged_raw_action=step or None,
                                     distance_cm=float(trace["distance_m"][step-1]*100) if step else None))
        fig.suptitle(f"Episode {ep}: actual robot observations alongside goal-distance curves\n{REASONS.get(ep, 'Selected paired case')}",
                     fontsize=15, y=.985)
        fig.text(.5, .012, "Rows: A learned dynamics + latent cost; B simulator dynamics + latent cost; C simulator dynamics + task criterion.\n"
                 "Images use shared raw-action times, except the last column, which shows each controller's own stopping point.",
                 ha="center", fontsize=9, color=".3")
        for suffix in ("png", "pdf"):
            fig.savefig(output / f"episode_{ep}_inspection.{suffix}", dpi=170, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    return manifest


HTML = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><title>Reach-Wall paired episode inspection</title>
<style>
*{box-sizing:border-box}body{font:16px/1.5 system-ui,sans-serif;color:#18202b;max-width:1360px;margin:24px auto;padding:0 20px;background:#fafbfc}
h1{margin-bottom:6px}p{margin:8px 0}.muted{color:#596272}.controls{display:flex;align-items:center;gap:12px;flex-wrap:wrap;background:white;border:1px solid #dce1e7;border-radius:8px;padding:16px;margin:18px 0}
button,select{font:inherit;padding:5px 12px;cursor:pointer}input[type=range]{flex:1;min-width:220px}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}.card{background:white;border:1px solid #dce1e7;border-top:4px solid var(--color);padding:14px;border-radius:8px}
.card h2{margin:0;font-size:21px}.card video{width:100%;display:block;background:#eee}.metric{font-variant-numeric:tabular-nums;font-size:20px;margin:8px 0}.status{min-height:48px;font-size:14px;color:#596272}.stopped{color:#21623d;font-weight:600}figure{margin:20px 0;background:white;border:1px solid #dce1e7;padding:10px}figure img{width:100%}a{color:#245e9d}#actionLabel{min-width:105px;font-variant-numeric:tabular-nums}
@media(max-width:750px){.cards{grid-template-columns:1fr}.card video{max-width:300px;margin:auto}}
</style>
<h1>Reach-Wall: inspect paired actual executions</h1>
<p>Each episode shares its initial state and goal across A/B/C. Move the slider to compare observations after the same raw simulator action.</p>
<p class="muted">Videos contain actual execution only. After a controller stops, its terminal frame is held and explicitly labelled; no later execution is implied. Playback speed is for viewing, not simulator time.</p>
<div class="controls"><label>Episode <select id="episode"></select></label><button id="play">Play</button><button id="back" aria-label="Previous action">−1</button><button id="forward" aria-label="Next action">+1</button><input id="step" aria-label="Executed raw simulator action" type="range" min="0" max="100" step="1" value="0"><strong id="actionLabel">Action 0 / 100</strong><label>Speed <select id="speed"><option value="5">5 actions/s</option><option value="10" selected>10 actions/s</option><option value="30">30 actions/s</option></select></label></div>
<p id="reason"></p><div class="cards" id="cards"></div>
<figure><a id="curveLink" target="_blank"><img id="curves" alt="Actual paired goal-distance curves, full range and near-goal detail"></a><figcaption>5 cm is the native success threshold. Dotted gray lines mark replanning starts; a controller has no further planning calls after stopping.</figcaption></figure>
<p><a id="sheetLink" target="_blank">Open frame contact sheet with curves</a> · <a id="pdfLink" target="_blank">PDF</a> · <a href="inspection_notes.md">Case metrics and interpretation limits</a></p>
<script id="inspectionData" type="application/json">__DATA__</script>
<script>
const data=JSON.parse(document.getElementById('inspectionData').textContent);
const episode=document.getElementById('episode'),step=document.getElementById('step');
const play=document.getElementById('play'),speed=document.getElementById('speed');
let running=false,baseTime=0,baseStep=0,lastStep=-1;
for(const ep of data.episodes){const option=document.createElement('option');option.value=ep.episode;option.textContent=ep.episode;episode.appendChild(option)}
const selected=()=>data.episodes.find(ep=>ep.episode===Number(episode.value));
function pause(){running=false;play.textContent='Play'}
function render(force=false){
 const action=Number(step.value),entry=selected();
 if(!force&&action===lastStep)return;lastStep=action;
 document.getElementById('actionLabel').textContent=`Action ${action} / 100`;
 for(const [method,trace] of Object.entries(entry.methods)){
  const frame=Math.min(action,trace.actions),video=document.getElementById(`video${method}`);
  const requested=(frame+.1)/trace.fps;
  if(video.readyState>=1){video.currentTime=Math.min(requested,Math.max(0,video.duration-.001));}
  const text=frame===0?'Initial observation · distance not logged':`Distance: ${(trace.distance_m[frame-1]*100).toFixed(2)} cm`;
  document.getElementById(`metric${method}`).textContent=text;
  const status=document.getElementById(`status${method}`);
  status.className='status'+(frame===trace.actions?' stopped':'');
  if(action>trace.actions){status.textContent=`Stopped at action ${trace.actions} (${trace.success?'success':'budget'}). Showing terminal frame; no later execution.`;}
  else if(action===trace.actions){status.textContent=`Stopped here at action ${frame}: ${trace.success?'native success':'action budget exhausted'}.`;}
  else{status.textContent=frame===0?'Before the first raw action.':`Actual observation after action ${frame}.`;}
 }
}
function selectEpisode(){
 pause();step.value=0;lastStep=-1;
 const entry=selected();document.getElementById('reason').textContent=entry.reason;
 document.getElementById('cards').innerHTML=Object.entries(entry.methods).map(([method,trace])=>`<section class="card" style="--color:${trace.color}"><h2>${method}</h2><p>${trace.label}</p><video id="video${method}" muted playsinline preload="auto" src="${trace.video}" aria-label="Controller ${method} actual execution"></video><div class="metric" id="metric${method}"></div><div class="status" id="status${method}"></div><a href="${trace.video}" target="_blank">Open original video</a></section>`).join('');
 for(const method of Object.keys(entry.methods)){document.getElementById(`video${method}`).addEventListener('loadeddata',()=>render(true));}
 document.getElementById('curves').src=`episode_${entry.episode}_curves.png`;
 document.getElementById('curveLink').href=`episode_${entry.episode}_curves.png`;
 document.getElementById('sheetLink').href=`episode_${entry.episode}_inspection.png`;
 document.getElementById('pdfLink').href=`episode_${entry.episode}_inspection.pdf`;
 render(true);
}
function tick(now){if(running){const action=Math.min(100,baseStep+Math.floor((now-baseTime)/1000*Number(speed.value)));step.value=action;render();if(action===100)pause();}requestAnimationFrame(tick);}
episode.addEventListener('change',selectEpisode);step.addEventListener('input',()=>{pause();render();});
play.addEventListener('click',()=>{if(running){pause();return;}if(Number(step.value)===100)step.value=0;baseStep=Number(step.value);baseTime=performance.now();running=true;play.textContent='Pause';});
speed.addEventListener('change',()=>{baseStep=Number(step.value);baseTime=performance.now();});
for(const [id,delta] of [['back',-1],['forward',1]])document.getElementById(id).addEventListener('click',()=>{pause();step.value=Math.min(100,Math.max(0,Number(step.value)+delta));render();});
selectEpisode();requestAnimationFrame(tick);
</script></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--episodes", type=int, nargs="+", default=[0, 14, 18])
    args = parser.parse_args()
    if len(set(args.episodes)) != len(args.episodes) or any(ep not in range(48) for ep in args.episodes):
        parser.error("Episodes must be unique integers between 0 and 47")
    runs = args.runs_dir.resolve()
    output = (args.output_dir or runs.parent / "analysis/paired_episodes").resolve()
    assets = output / "videos"
    assets.mkdir(parents=True, exist_ok=True)
    all_traces, _, all_sources = load_traces(runs)
    metrics, frames_manifest, source_videos, selected_sources, long_rows, data = [], [], {}, {}, [], []
    for ep in args.episodes:
        records = {method: next(t for t in values if t["episode"] == ep) for method, values in all_traces.items()}
        starts, frames, view = {}, {}, {}
        for method, trace in records.items():
            rows, starts[method], video = execution_records(runs, method, ep)
            actions = int(trace["steps"][-1])
            copied = assets / f"episode_{ep}_{method}.mp4"
            frames[method], source_videos[f"{ep}/{method}"] = copy_video_and_frames(video, copied, actions)
            rel = str((video.parent.parent if method == "A" else video.parent) / "execution.csv")
            rel = str(Path(rel).relative_to(runs))
            selected_sources[rel] = all_sources[rel]
            distance = trace["distance_m"]
            metrics.append(dict(episode=ep, method=method, success=bool(trace["success"][-1]),
                actual_actions=actions, final_distance_cm=float(distance[-1]*100),
                minimum_distance_cm=float(distance.min()*100), minimum_at_action=int(np.argmin(distance)+1),
                distance_at_action_30_cm=float(distance[29]*100) if actions >= 30 else None,
                distance_at_action_90_cm=float(distance[89]*100) if actions >= 90 else None,
                improvement_actions_90_to_100_cm=float((distance[89]-distance[99])*100) if actions == 100 else None,
                replanning_starts=";".join(map(str, starts[method]))))
            for index, row in enumerate(rows):
                long_rows.append(dict(episode=ep, method=method, raw_action=int(row["simulator_step"]),
                    plan_call=int(row["plan_call"]), tcp_goal_distance_m=float(row["tcp_goal_distance_m"]),
                    native_success=row["success"] == "True"))
                if float(row["tcp_goal_distance_m"]) != distance[index]:
                    raise AssertionError("Execution changed during inspection")
            view[method] = dict(video=f"videos/{copied.name}", fps=30, actions=actions,
                distance_m=distance.tolist(), success=bool(trace["success"][-1]),
                color=METHODS[method]["color"], label=METHODS[method]["label"])
        frames_manifest.extend(figures(ep, records, starts, frames, output))
        data.append(dict(episode=ep, reason=REASONS.get(ep, "Selected paired case"), methods=view))
    write_csv(output / "paired_episode_metrics.csv", metrics)
    write_csv(output / "distance_traces.csv", long_rows)
    write_csv(output / "frame_manifest.csv", frames_manifest)
    viewer = dict(episodes=data)
    (output / "viewer_data.json").write_text(json.dumps(viewer, indent=2)+"\n", encoding="utf-8")
    (output / "index.html").write_text(HTML.replace("__DATA__", json.dumps(viewer)), encoding="utf-8")
    protocol = dict(episodes=args.episodes, case_selection={str(ep):REASONS.get(ep,"User-selected") for ep in args.episodes},
        source_execution_csv_sha256=selected_sources, source_videos=source_videos,
        simulator_replay=False, model_inference=False, distance_source="Native TCP-to-goal distance in actual execution.csv",
        figure_policy="No values after stopping. Last contact-sheet column uses each method's own stopping action.",
        viewer_policy="Shared raw-action slider; after stopping hold only the labelled terminal video frame.",
        interpretation="Outcome-selected illustrative cases, not a random sample or causal test. Initial state/goal are paired; later controller states differ.")
    (output / "protocol.json").write_text(json.dumps(protocol,indent=2)+"\n", encoding="utf-8")
    notes = ["# Paired Reach-Wall episode inspection", "", "Actual execution only; distances in cm. No simulator or model rerun.", "",
        "| Episode | Selection | A final | B final | C final | Native successes A/B/C |", "|---|---|---:|---:|---:|---|"]
    for ep in args.episodes:
        by = {r["method"]:r for r in metrics if r["episode"] == ep}
        notes.append(f"| {ep} | {REASONS.get(ep,'Selected')} | {by['A']['final_distance_cm']:.2f} | {by['B']['final_distance_cm']:.2f} | {by['C']['final_distance_cm']:.2f} | " + "/".join("yes" if by[m]["success"] else "no" for m in METHODS) + " |")
    notes.extend(["", "## Reading the figures and viewer", "",
        "- Open index.html to scrub all three actual videos at the same raw action. The controls also permit slow playback and single-action steps.",
        "- Native success is TCP distance <=5 cm. Dotted plot lines indicate replanning starts, not CEM rounds.",
        "- Curves stop with the actual episode. The viewer holds and labels a stopped terminal frame; contact sheets leave later shared times blank.",
        "- Frame 0 is the initial observation; no step-0 distance was logged. Frame k follows raw action k, verified from video length and recording code.",
        "- The contact sheet's last column uses each method's own stopping time, unlike its preceding shared-time columns.", "",
        "## Interpretation limits", "",
        "These cases were chosen using observed outcomes. They illustrate behavior but do not estimate its frequency or establish why a controller fails.",
        "The same initial state and goal are paired across controllers. Their later visited states, candidate populations, and selected actions differ.",
        "Images are lossy 224x224 camera observations. Do not infer exact 3D TCP coordinates, contact, or a collision from apparent image overlap.",
        "A's successful episode 14 makes its large improvement during the final call. The saved controllers shorten lookahead near the action budget (30, then 25, then 10 raw actions). Coincidence with late improvement is a hypothesis prompt, not a demonstrated mechanism."])
    (output / "inspection_notes.md").write_text("\n".join(notes)+"\n", encoding="utf-8")
    print(json.dumps(metrics,indent=2))
    print(f"Saved {len(args.episodes)} paired cases, {len(metrics)} actual videos, and offline viewer: {output / 'index.html'}")


if __name__ == "__main__":
    main()

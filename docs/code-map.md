# Code and provenance

The `jepa-wms/` directory contains a source snapshot of
[`facebookresearch/jepa-wms`](https://github.com/facebookresearch/jepa-wms) at
commit `13cf1d9c7e476f53c17714d2e0f1dc239a883ce0`, with this study's extensions.
The 316 baseline files retain their upstream attribution and notices. The baseline
includes training/model code, but this project uses released pretrained models.

[`data/manifest.json`](../data/manifest.json) lists every copied source, hash, added
study file and modified upstream module. The original tracked changes are also
available as [`environment/upstream-changes.patch`](../environment/upstream-changes.patch).
The Git history of the original repository and its credentials are not copied.

## Main reading path

Paths below are relative to `jepa-wms/`.

| Purpose | File / function |
| --- | --- |
| Plan, act, observe and repeat | `evals/simu_env_planning/planning/plan_evaluator.py`: `PlanEvaluator.unroll_agent` |
| CEM sampling, bounding, elites and final-mean return | `evals/simu_env_planning/planning/planning/planner.py`: `CEMPlanner.plan` |
| Latent goal score | `evals/simu_env_planning/planning/planning/objectives.py`: `ReprTargetDistMPCObjective` |
| Released encoder and predictive dynamics | `app/vjepa_wm/modelcustom/simu_env_planning/vit_enc_preds.py`: `EncPredWM.encode`, `unroll` |
| Native Meta-World actions, success and state restoration | `evals/simu_env_planning/envs/metaworld.py`: `MetaWorldWrapper` |

## Added study modules

All planning module names below live under `evals/simu_env_planning/planning/`.

| Experiment or support | Modules |
| --- | --- |
| Candidate logging with action sequences and physics snapshots | `candidate_logging.py` |
| Wall simulator replay, latent and task scoring | `candidate_replay.py` (historical executable, not an import-safe helper) |
| Wall ranking, regret and round analysis | `analyze_candidates.py`, `analyze_search_budget.py`, `plot_candidates.py` |
| Wall matched-terminal control and route inspection | `analyze_wall_terminal.py`, `inspect_wall_routes.py`, `plot_wall_pool_routes.py` |
| Reach-Wall shared-pool B/C | `reach_wall_b.py`, `reach_wall_c.py`, `run_reach_wall_bc.py` |
| Reach-Wall matched-terminal control | `analyze_reach_wall_terminal.py` |
| Independent oracle-dynamics / latent-score CEM controller | `reach_wall_b_closed_loop.py`: `OracleEndpointUnroll`, `run_episode` |
| Independent native task-score CEM controller | `reach_wall_c_closed_loop.py`: `OracleTaskUnroll`, `TaskObjective`, `run_episode` |
| Native distance curves and paired episodes | `plot_reach_wall_distance.py`, `inspect_reach_wall_episodes.py` |
| Selected prefix versus complete horizon | `analyze_reach_wall_prefix.py` |
| New-prefix coverage, elite membership and score audit | `replay_reach_wall_prefix_candidates.py`, `analyze_reach_wall_prefix_candidates.py` |
| Matched-initial-state independent search | `replay_reach_wall_initial_search.py`, `analyze_reach_wall_initial_search.py` |

Additional support modules are `evals/simu_env_planning/episode_resume.py` and
`resource_monitor.py`. Exact experiment configs are
`configs/evals/simu_env_planning/wall/jepa-wm/pool650_i10.yaml` and
`configs/evals/simu_env_planning/mw/jepa-wm/reach_wall_a.yaml`.

## Modified upstream paths

The study modifies planner batching/capture/action-bound handling; evaluator and
agent capture/stopping; Meta-World snapshot/normalization/native-success handling;
resume support; and Windows-compatible distributed/cluster/video setup. The exact
12 modified paths are listed in the manifest and patch. These changes are opt-in
or task-specific where appropriate; unchanged upstream defaults are not claimed
as implementations of this study's protocol.

## Tests and release checks

Scientific edge-case tests cover tied task ranks, distinct deployed means versus
minimum-score candidates, conditional regret units, first-hit versus endpoint
semantics, reached-then-left routes, inclusive native success, duplicate routes and
undefined constant correlations. Additional native tests cover snapshot restoration,
bounded actions, split scoring batches and resumability; some require simulator
dependencies and original run artifacts. The lightweight CI subset is documented
in [reproduction](reproduction.md).

Portfolio additions outside the vendored source are `scripts/check_results.py`
(independent reconciliation of released measurements), `scripts/plot_results.py`
(data-only figure regeneration), protocol/reproduction docs, selected figures and
the public data manifest.

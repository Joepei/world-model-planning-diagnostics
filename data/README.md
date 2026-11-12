# Published data

Small, frozen measurements and derived tables are included so the main claims and figures can be checked without downloading model weights. CSV columns retain numeric values; distances marked `_m` are metres and `_cm` are centimetres. Wall coordinates are their own simulator units.

| Directory | Unit of observation | Contents |
| --- | --- | --- |
| `wall/scored_candidates.csv` | One captured candidate occurrence | All 6,500 A/B scores and first-hit/terminal outcomes |
| `wall/first_hit/` | Episode or CEM round | Original task-order summaries |
| `wall/terminal/` | Episode or selection | Endpoint control, regret, rank agreement |
| `wall/routes/` | Unique episode/route pair | All 6,410 native 31-position trajectories, IDs, geometry |
| `reach_wall/shared_pool/{first_hit,terminal}/` | A decision state or selected route | 336 pools; per-pool correlations and selections |
| `reach_wall/executions/{A,B,C}/` | One executed raw action | All 144 native episode CSVs |
| `reach_wall/closed_loop/` | Executed action or episode | Consolidated 11,109 action outcomes and 144 episode summaries |
| `reach_wall/prefix_vs_full/` | Selected plan | 672 A/B selected-plan outcomes and aggregates |
| `reach_wall/b_prefix_candidates/` | Audited B state or CEM round | 47 successor-state pools and round summaries |
| `reach_wall/initial_search_ab/` | Matched initial state or search round | Independent A/B search comparisons at 48 common starts |

The full 1.5 GB Reach-Wall fixed-pool score database, hundreds of megabytes of auxiliary native caches, `.pt` simulator/candidate bundles, model weights and videos are not included. Their absence means the complete candidate-level Reach-Wall ranking analysis cannot be reconstructed from this lightweight release alone. The published per-pool summaries support aggregate reproduction; the bundled capture/replay code supports generating new full data. There is no promise of identical new trajectories across hardware, simulator versions, renderers and search seeds.

`manifest.json` records original source hashes, published hashes, upstream provenance, and the exact copied files. For JSON metadata copies, local absolute project paths are replaced with `<PROJECT>`; original source hashes and the transformation are retained. Measurements are not changed. Root figure generation adds a separate `figures/manifest.json` recording its inputs.

Run `python scripts/check_results.py` to check the published source hashes, native traces, terminal summary arithmetic, route-cache endpoints, and headline counts. Read [the protocol](../docs/experiments.md) before comparing different units or C criteria.

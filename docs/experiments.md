# Experimental protocol

## What A/B/C identify

In a shared pool, A/B differ in how future endpoint embeddings are obtained. Both use the same encoder, goal, latent distance and candidate actions. This measures evaluation differences conditional on A's search and visited states. B minimizes the exact simulator-endpoint latent score; that does not guarantee minimization of physical goal distance. B/C disagreement combines the chosen feature metric, modality weights, goal formulation and, for first-hit C, temporal preferences. It does not establish missing information in the encoder.

In independent control, each objective guides its own CEM updates and thus candidate generation and future state visitation. Later A/B states are not matched. The first CEM population at each paired episode's initial state is matched; later populations within that first planning call already diverge.

## Wall

- Ten episodes, seed 1; one open-loop planning call per episode.
- CEM: 10 rounds × 64 evaluated samples; ten elites; one separately scored updated mean per round.
- Horizon: six model steps with five raw commands packed per step = 30 two-dimensional actions.
- A executes the final elite mean, not necessarily the lowest-cost sample.
- Pool: 650 occurrences, 641 unique routes per episode. Duplicates are collapsed within the same state; exact score ties receive average ranks.
- A/B: full-horizon visual/proprioceptive latent endpoint cost, visual MSE + 0.1 × proprioceptive MSE.
- Native first-hit success: position distance strictly <4.5 Wall coordinate units.
- Original C: success before failure; earlier first-hit action among successes; doorway-aware remaining geometric distance among failures.
- Terminal C: continuous Euclidean distance at the diagnostic action-30 endpoint for every route; no arrival-time or doorway-distance term. Terminal success <4.5 is a separate diagnostic.
- The original Gaussian search was not clipped to the declared [-1,1]^2 action box. Some saved actions exceed it. These are historical results, not results from a bounded action-space benchmark.

The terminal control changes both timing preferences and the failure-distance geometry. Its correlation difference cannot be attributed solely to timing. Of 627 unique first-hit-successful routes, 357 finish inside the goal and 270 leave. Episodes 0, 3, and 4 contain no successful route. Episodes 3 and 4 account for 98.64% of B's total terminal-distance regret; all ten cases share wall geometry but differ in start and goal positions.

Primary data: [occurrence scores](../data/wall/scored_candidates.csv), [terminal results](../data/wall/terminal/), [native route cache](../data/wall/routes/).

## Reach-Wall shared pools

- 48 episodes, seed 1; seven A planning calls per episode, at raw actions 0/15/30/45/60/75/90.
- CEM: 15 rounds × 300 evaluated samples, including the inserted distribution mean; ten elites; one updated mean per round.
- Each decision contains 4,515 occurrences / 4,501 unique full routes; total 1,517,040 / 1,512,336.
- Nominal horizon six packed model steps = 30 raw four-dimensional commands. Prefix three model steps = 15 commands. Raw bounds [-1,1]^4 are enforced using the training-statistics normalization transform.
- Full horizons are 30/30/30/30/30/25/10 actions as the episode budget is consumed. Prefix limits are 15/15/15/15/15/15/10.
- Native success: gripper-centre/TCP distance <=0.05 m, using simulator measurements rather than the expert-proprio diagnostic.
- A/B cost: endpoint visual MSE + 0.1 × proprioceptive-embedding MSE.
- Primary C-terminal: continuous native TCP distance at the same complete candidate endpoint. Original C-first-hit remains separately available.
- B diagnostic rollouts continue after an earlier hit so its endpoint can match A's horizon. First-hit and prefix outcomes are recorded separately. Actual episodes stop at native success.
- Full-horizon terminal successes for deployed A / minimum A / minimum B / terminal best: 14/13/35/243 out of 336. Original first-hit counts: 14/14/36/243.
- Terminal prefix successes: 6/6/7/103. A/B still select using full-horizon scores; this prefix view does not match their scoring time.

All routes are restored to the same saved MuJoCo integration state, model values, observation history, native RNG, action clock and goal for each decision. State/native-prefix checks and source hashes are retained. Sparse OpenGL and floating-point differences are handled separately from native-physics validation.

Primary data: [terminal tables](../data/reach_wall/shared_pool/terminal/), [original first-hit tables](../data/reach_wall/shared_pool/first_hit/).

## Reach-Wall independent closed-loop controllers

The three controllers start from the same 48 saved states, goals and initial sample populations. Each then searches and acts independently with the same search budget, native bounds, final-mean return rule, and 100-action budget. Initialization warmup is excluded from that budget. These opt-in protocol choices mean the study is not an exact reproduction of the released evaluation.

A uses learned dynamics and endpoint latent cost. B simulates its own CEM candidates, encodes their complete endpoints, and uses the same latent cost. C simulates its own candidates and orders them by **first-hit task performance**, rather than the shared-pool terminal control: success first, then earlier native first-hit action; failed candidates rank by terminal TCP distance. Equal first-hit times tie. Its numeric CEM key only preserves ordering and is not a physical mixed-unit loss.

Actual execution stops at first success for all controllers. C candidate task rollouts stop at first hit; B scoring continues through the full horizon. All use final elite means. There is no carryover/warm start of the old remainder into the next search.

Native episode successes: A 6/48, B 0/48, C 48/48. Mean final distances: 6.81071, 6.20374, 4.81946 cm. A succeeds in episodes 14, 31, 34, 36, 42 and 46. A executes 4,799 total actions, B 4,800, C 1,510. Each curve stops at its actual episode endpoint.

Across these different starting states, B's final-distance standard deviation is 0.553 cm versus A's 1.426 cm. This is not repeated search-seed variability at a fixed state. The A late improvement coincides with both a shorter horizon and full execution of the remaining plan; neither effect is isolated here.

Primary data: [144 native execution logs](../data/reach_wall/executions/), [distance traces and episode summaries](../data/reach_wall/closed_loop/).

## Selected-plan and successor-state audits

The nominal 30-action subset contains 240 selected plans per controller, from the first five calls. All native prefix successes are zero; full-plan successes are A 8 and B 47. B's 47 successes occur in 43 episodes; only four episodes contain consecutive successful full-plan calls. This is not continual renewal of a successful full plan at every call.

The deeper audit is conditioned on the 47 B successor states following a successful full-plan continuation. At all 47 states the old 15-action remainder still succeeds. New pools contain successful 15-action prefixes in 44 cases, yet final elites and means contain none. Of 211,500 evaluated sample occurrences, 3,303 succeed within the prefix; one additional intermediate updated mean does. Only 42 successful sampled prefixes ever enter their own round's elite set. One pool contains a successful prefix cheaper than the returned mean, but its lowest-cost failed-prefix sample remains cheaper.

The previous remainder is a feasibility reference, not an inserted candidate with an invented 30-action score. Exact action repetition in continuous space is not the substantive coverage test. Of 3,304 successful-prefix occurrences, 1,850 also end inside the goal after 30; 1,454 leave. In 41 states, an available prefix both succeeds early and ends in the goal while the returned mean's full endpoint is outside. Thus earlier-hit versus endpoint timing is insufficient as a complete explanation.

Native scoring is available for every captured prefix; full outcomes were additionally checked for successful prefixes, elites, updated means and minimum-score samples. Other full native outcomes in this audit remain unavailable. The independent initial-search study below, in contrast, replays every B full route.

Primary data: [selected-plan audit](../data/reach_wall/prefix_vs_full/), [successor-state audit](../data/reach_wall/b_prefix_candidates/).

## Matched initial-state search

At all 48 initial states, A/B share the same goal, native state, first 300 samples, 30-action horizon, 15-action prefix and 15×300 CEM budget. Each method contributes 216,000 evaluated sample occurrences across episodes. All 14,400 shared first-population routes have matching native outcomes with zero observed distance error.

B produces 792 successful sampled full routes, all with successful action-30 endpoints, concentrated in four initial-state pools (episodes 5, 11, 15, 20). A has zero. B returns two successful full means. Neither method produces a successful first-15-action prefix at these initial states. When both full pools fail (44 starts), A's pool best is closer in 28 and B's in 16; four additional B wins arise from successful full routes. These metrics do not establish an overall dominance relation.

Equal proposal budget is not equal runtime. Pool best is relative to a finite observed pool, not the space of all admissible sequences. Published exploratory bootstrap intervals resample 48 episode IDs rather than adaptive candidate occurrences; they do not replace repeated search seeds.

Primary data: [matched initial-state tables](../data/reach_wall/initial_search_ab/).

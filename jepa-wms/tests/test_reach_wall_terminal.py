"""Scientific contracts for matched-terminal versus first-hit evaluation."""

import unittest

from evals.simu_env_planning.planning.analyze_reach_wall_terminal import analyze_pool, aggregate
from tests.test_reach_wall_c import candidate


def pool():
    # c0 arrives earliest but leaves; c2 arrives later and has the best endpoint.
    rows = [candidate(0, True, 10, .04, a=.1, b=2.),
            candidate(1, True, 20, .04, a=.8, b=.7),
            candidate(2, True, 25, .045, a=1.2, b=.6),
            candidate(3, distance=.12, a=.9, b=1.)]
    for row, endpoint, prefix in zip(rows, (.15, .03, .02, .12), (.04, .09, .1, .11)):
        row.update(full_horizon_goal_distance_m=endpoint, full_horizon_success=endpoint <= .05,
                   prefix_horizon_goal_distance_m=prefix)
    return rows


class ReachWallTerminalTests(unittest.TestCase):
    def test_reached_then_left_is_not_a_terminal_success(self):
        ranked, summary, selected = analyze_pool(pool(), "full_horizon")
        choices = {row["selection"]: row for row in selected}
        early = next(row for row in ranked if row["candidate"] == "c0")
        self.assertTrue(early["c_success"])
        self.assertFalse(early["terminal_success"])
        self.assertEqual(choices["c_first_hit_best"]["candidate"], "c0")
        self.assertEqual(choices["c_terminal_best"]["candidate"], "c2")
        self.assertEqual(summary["reached_then_left_routes"], 1)

    def test_successes_rank_by_endpoint_not_arrival_time(self):
        ranked, _, _ = analyze_pool(pool(), "full_horizon")
        lookup = {row["candidate"]: row for row in ranked}
        self.assertLess(lookup["c1"]["c_rank"], lookup["c2"]["c_rank"])
        self.assertGreater(lookup["c1"]["terminal_rank"], lookup["c2"]["terminal_rank"])
        rows = pool()
        rows[1]["full_horizon_goal_distance_m"] = .02
        ranked, _, _ = analyze_pool(rows, "full_horizon")
        lookup = {row["candidate"]: row for row in ranked}
        self.assertEqual(lookup["c1"]["terminal_rank"], lookup["c2"]["terminal_rank"])

    def test_native_threshold_is_inclusive(self):
        rows = pool()
        rows[1].update(full_horizon_goal_distance_m=.05, full_horizon_success=True)
        ranked, _, _ = analyze_pool(rows, "full_horizon")
        self.assertTrue(next(row for row in ranked if row["candidate"] == "c1")["terminal_success"])
        rows[1]["full_horizon_success"] = False
        with self.assertRaisesRegex(ValueError, "Native full-horizon"):
            analyze_pool(rows, "full_horizon")

    def test_actual_mean_and_latent_selections_remain_distinct(self):
        _, _, selected = analyze_pool(pool(), "full_horizon")
        choices = {row["selection"]: row for row in selected}
        self.assertEqual(choices["a_deployed_mean"]["candidate"], "c3")
        self.assertEqual(choices["a_pool_min"]["candidate"], "c0")
        self.assertEqual(choices["b_pool_min"]["candidate"], "c2")
        self.assertAlmostEqual(choices["a_deployed_mean"]["terminal_regret_m"], .10)
        self.assertAlmostEqual(choices["a_pool_min"]["terminal_regret_m"], .13)

    def test_prefix_uses_its_unstopped_endpoint_and_same_latent_choice(self):
        rows = pool()
        rows[0]["prefix_horizon_goal_distance_m"] = .08
        ranked, _, selected = analyze_pool(rows, "executed_prefix")
        early = next(row for row in ranked if row["candidate"] == "c0")
        self.assertTrue(early["c_success"])
        self.assertFalse(early["terminal_success"])
        choices = {row["selection"]: row for row in selected}
        self.assertEqual(choices["b_pool_min"]["candidate"], "c2")
        self.assertFalse(choices["b_pool_min"]["first_hit_success"])

    def test_constant_latent_rank_is_undefined_not_zero(self):
        rows = pool()
        for row in rows:
            row["a_cost"] = 1.
        _, summary, _ = analyze_pool(rows, "full_horizon")
        self.assertIsNone(summary["spearman_a_terminal"])

    def test_duplicate_routes_do_not_inflate_coverage(self):
        rows = pool()
        rows.append(dict(rows[2], candidate_index=4, candidate="duplicate"))
        ranked, summary, _ = analyze_pool(rows, "full_horizon")
        self.assertEqual(len(ranked), 4)
        self.assertEqual(summary["occurrences"], 5)
        self.assertEqual(summary["terminal_success_routes"], 2)

    def test_aggregation_keeps_first_hit_and_terminal_separate(self):
        pools, selections = [], []
        for scope in ("full_horizon", "executed_prefix"):
            _, summary, selected = analyze_pool(pool(), scope)
            pools.append(summary)
            selections.extend(selected)
        result = aggregate(pools, selections)
        self.assertEqual(result["completed_pools"], 1)
        full = result["scopes"]["full_horizon"]
        self.assertEqual(full["selections"]["a_pool_min"]["first_hit_successes"], 1)
        self.assertEqual(full["selections"]["a_pool_min"]["terminal_successes"], 0)


if __name__ == "__main__":
    unittest.main()

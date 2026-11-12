"""Scientific controls for terminal versus stop-on-first-hit analysis."""

import unittest

from evals.simu_env_planning.planning.analyze_wall_terminal import (
    analyze_episode, correlations, rank_terminal_candidates,
)


def route(name, distance, success=False, hit=None, a=1.0, b=1.0, **extra):
    row = dict(
        episode=0, candidate=name, route_id=name, cem_round=1,
        a_cost=a, b_cost=b, end_distance=distance,
        c_success=success, c_first_hit_step=hit,
        c_remaining_distance=2.0 if success else distance + 1,
        task_end_distance=2.0 if success else distance, is_deployed_output=False,
    )
    row.update(extra)
    return row


class TestWallTerminal(unittest.TestCase):
    def test_first_hit_winner_can_lose_terminal_comparison(self):
        rows = [route("early_then_left", 10, success=True, hit=2, b=2),
                route("late_and_inside", 1, success=True, hit=20, b=1)]
        ranked = rank_terminal_candidates(rows)
        self.assertEqual([row["c_rank"] for row in ranked], [1, 2])
        self.assertEqual([row["terminal_rank"] for row in ranked], [2, 1])
        self.assertEqual([row["terminal_success"] for row in ranked], [False, True])
        self.assertAlmostEqual(correlations(ranked)["spearman_b_terminal"], 1)
        self.assertAlmostEqual(correlations(ranked)["spearman_b_first_hit"], -1)

    def test_threshold_is_strict_and_terminal_ties_are_averaged(self):
        ranked = rank_terminal_candidates([route("one", 4.5), route("two", 4.5),
                                           route("inside", 4.49, success=True, hit=30)])
        self.assertEqual([row["terminal_rank"] for row in ranked], [2.5, 2.5, 1])
        self.assertEqual([row["terminal_success"] for row in ranked], [False, False, True])

    def test_terminal_success_without_first_hit_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "first-hit"):
            rank_terminal_candidates([route("inconsistent", 1)])

    def test_constant_terminal_ranking_is_undefined(self):
        ranked = rank_terminal_candidates([route("one", 10, b=1), route("two", 10, b=2)])
        self.assertIsNone(correlations(ranked)["spearman_b_terminal"])

    def test_regret_uses_full_endpoint_and_preserves_deployed_alias(self):
        first = route("early_then_left", 10, success=True, hit=2, a=2, b=1)
        alias = dict(first, candidate="actual_final_mean", a_cost=2.0000001, is_deployed_output=True)
        late = route("terminal_best", 1, success=True, hit=20, a=1, b=2)
        ranked, summary, selections = analyze_episode([first, alias, late])
        self.assertEqual(len(ranked), 2)
        self.assertEqual(summary["first_hit_success_terminal_failure_routes"], 1)
        selected = {row["selection"]: row for row in selections}
        self.assertEqual(selected["deployed"]["candidate"], "actual_final_mean")
        self.assertEqual(selected["min_terminal"]["candidate"], "terminal_best")
        self.assertEqual(selected["min_first_hit"]["candidate"], "early_then_left")
        self.assertEqual(selected["min_b"]["terminal_distance_regret"], 9)
        self.assertEqual(selected["min_b"]["terminal_missed_success"], 1)
        self.assertTrue(selected["min_b"]["first_hit_optimal"])
        self.assertFalse(selected["min_b"]["terminal_optimal"])
        self.assertEqual(selected["min_terminal"]["terminal_distance_regret"], 0)

    def test_distinct_routes_with_equal_terminal_minimum_are_all_optimal(self):
        _, _, selections = analyze_episode([
            route("deployed", 5, a=1, b=2, is_deployed_output=True),
            route("b_best", 5, a=2, b=1),
        ])
        self.assertTrue(all(row["terminal_optimal"] for row in selections))
        self.assertTrue(all(row["terminal_distance_regret"] == 0 for row in selections))

    def test_original_c_tied_optima_can_have_different_terminal_outcomes(self):
        _, summary, _ = analyze_episode([
            route("stayed_inside", 1, success=True, hit=3, is_deployed_output=True),
            route("left_after_hit", 10, success=True, hit=3),
        ])
        self.assertEqual(summary["original_c_optimal_route_count"], 2)
        self.assertEqual(summary["original_c_optimal_terminal_success_routes"], 1)
        self.assertEqual(summary["original_c_optimal_terminal_distance_min"], 1)
        self.assertEqual(summary["original_c_optimal_terminal_distance_max"], 10)

    def test_distinct_episodes_are_not_pooled(self):
        with self.assertRaisesRegex(ValueError, "one episode"):
            rank_terminal_candidates([route("one", 10), route("two", 12, episode=1)])


if __name__ == "__main__":
    unittest.main()

"""Scientific edge cases for fixed-pool ranking and regret."""

import unittest

from evals.simu_env_planning.planning.analyze_candidates import (
    deduplicate_routes,
    rank_candidates,
    rank_correlations,
    regret_against_pool,
    summarize_episode,
)


def route(name, a=1.0, b=1.0, success=False, hit=None, remaining=10.0, episode=0, **extra):
    row = {
        "episode": episode,
        "candidate": name,
        "route_id": name,
        "cem_round": 1,
        "a_cost": a,
        "b_cost": b,
        "c_success": success,
        "c_first_hit_step": hit,
        "c_remaining_distance": remaining,
        "task_end_distance": remaining,
        "end_distance": remaining,
        "is_deployed_output": False,
    }
    row.update(extra)
    return row


class TestCandidateAnalysis(unittest.TestCase):
    def test_success_beats_failure_and_equal_arrivals_tie(self):
        rows = [route("one", success=True, hit=4, remaining=3.0),
                route("two", success=True, hit=4, remaining=1.0),
                route("failure", remaining=4.6)]
        ranked = rank_candidates(rows)
        self.assertEqual([row["c_rank"] for row in ranked], [1.5, 1.5, 3.0])

    def test_dedup_keeps_first_score_and_deployed_alias(self):
        first = route("first", a=0.5)
        second = route("alias", a=0.5 - 4e-7, route_id="first", is_deployed_output=True)
        other_episode = route("first", episode=1)
        unique = deduplicate_routes([first, second, other_episode])
        self.assertEqual(len(unique), 2)
        self.assertEqual(unique[0]["a_cost"], 0.5)
        self.assertEqual(unique[0]["occurrence_count"], 2)
        self.assertTrue(unique[0]["contains_deployed_output"])

    def test_inconsistent_duplicate_outcomes_are_rejected(self):
        with self.assertRaises(ValueError):
            deduplicate_routes([route("one"), route("alias", route_id="one", b=2.0)])

    def test_spearman_compares_aligned_candidates(self):
        ranked = rank_candidates([route("one", a=1, b=3, remaining=1),
                                  route("two", a=2, b=2, remaining=2),
                                  route("three", a=3, b=1, remaining=3)])
        correlations = rank_correlations(ranked)
        self.assertAlmostEqual(correlations["spearman_ab"], -1.0)
        self.assertAlmostEqual(correlations["spearman_ac"], 1.0)

    def test_constant_rank_correlation_is_not_zero(self):
        ranked = rank_candidates([route("one", a=1, b=1), route("two", a=1, b=2)])
        self.assertIsNone(rank_correlations(ranked)["spearman_ab"])

    def test_regret_preserves_units_and_counts_c_ties_as_optimal(self):
        oracle = route("oracle", b=0.1, success=True, hit=3, remaining=4.0)
        failed = regret_against_pool(route("failed", b=0.5), oracle, oracle)
        self.assertEqual(failed["c_missed_success"], 1)
        self.assertIsNone(failed["c_excess_steps"])
        self.assertIsNone(failed["c_excess_distance"])
        late = regret_against_pool(route("late", b=0.5, success=True, hit=8, remaining=2.0), oracle, oracle)
        self.assertEqual(late["c_excess_steps"], 5)
        tied = regret_against_pool(route("tie", b=0.2, success=True, hit=3, remaining=1.0), oracle, oracle)
        self.assertTrue(tied["c_optimal"])
        failure_oracle = route("closest", b=0.1, remaining=4.0)
        failure_gap = regret_against_pool(route("far", b=0.5, remaining=7.0), failure_oracle, failure_oracle)
        self.assertEqual(failure_gap["c_excess_distance"], 3.0)
        self.assertIsNone(failure_gap["c_excess_steps"])

    def test_deployed_output_is_not_replaced_by_minimum_a(self):
        rows = [route("deployed", a=5.0, b=4.0, is_deployed_output=True),
                route("min_a", a=1.0, b=3.0),
                route("oracle", a=2.0, b=0.1, success=True, hit=2, remaining=2.0)]
        _, summary, regret = summarize_episode(rows)
        self.assertEqual(summary["deployed_candidate"], "deployed")
        self.assertEqual(summary["min_a_candidate"], "min_a")
        self.assertEqual(regret["deployed_c_missed_success"], 1)
        self.assertAlmostEqual(regret["deployed_b_regret"], 3.9)
        self.assertAlmostEqual(regret["deployed_a_return_gap"], 4.0)


if __name__ == "__main__":
    unittest.main()

"""Scoring batches must preserve the full CEM population, ordering, and RNG."""

import unittest

import torch

from evals.simu_env_planning.planning.planning.planner import CEMPlanner


class CEMBatchingTests(unittest.TestCase):
    def test_invalid_batch_size_is_rejected(self):
        for size in (0, -1, True, 1.5):
            with self.subTest(size=size), self.assertRaises(ValueError):
                CEMPlanner(lambda *args: None, cost_batch_size=size)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA planner")
    def test_full_population_matches_batched_costs_elites_and_random_stream(self):
        def run(batch_size):
            seen = []

            def unroll(z_init, act_suffix):
                seen.append(act_suffix.shape[1])
                return act_suffix.cumsum(dim=0)

            generator = torch.Generator(device="cuda").manual_seed(72)
            planner = CEMPlanner(
                unroll, iterations=2, num_samples=300, num_elites=10, horizon=3,
                action_dim=4, num_act_stepped=2, cost_batch_size=batch_size,
                local_generator=generator, capture_rounds=[1, 2], capture_samples=True,
            )
            planner.set_objective(lambda predicted, actions: predicted.square().sum(dim=(0, 2)))
            result = planner.plan(torch.zeros(1, device="cuda"))
            return result, generator.get_state(), seen

        full, full_rng, _ = run(None)
        batched, batched_rng, seen = run(32)
        self.assertLessEqual(max(seen), 32)
        self.assertIn(12, seen)  # Final partial batch: 300 = 9 * 32 + 12.
        self.assertTrue(torch.equal(full_rng, batched_rng))
        torch.testing.assert_close(full.actions, batched.actions, rtol=0, atol=0)
        first, second = full.info["diagnostic_candidates"], batched.info["diagnostic_candidates"]
        self.assertEqual(len(first), 2 * 301)
        self.assertEqual(len(first), len(second))
        for a, b in zip(first, second):
            self.assertEqual((a["name"], a["is_elite"], a["a_cost"]), (b["name"], b["is_elite"], b["a_cost"]))
            torch.testing.assert_close(a["actions_normalized"], b["actions_normalized"], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()

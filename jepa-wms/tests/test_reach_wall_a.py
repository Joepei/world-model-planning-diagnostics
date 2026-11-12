"""Checks for the scientific contracts needed by bounded Reach-Wall A."""

import unittest
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import torch

from evals.simu_env_planning.envs.init import make_env
from evals.simu_env_planning.planning.common.parser import parse_cfg
from evals.simu_env_planning.planning.planning.planner import CEMPlanner
from evals.utils import make_datasets


CONFIG = Path(__file__).resolve().parents[1] / "configs/evals/simu_env_planning/mw/jepa-wm/reach_wall_a.yaml"


class ReachWallContracts(unittest.TestCase):
    def setUp(self):
        self.cfg = OmegaConf.load(CONFIG)
        self.cfg.work_dir = "."
        self.cfg.frameskip = 5
        self.cfg = parse_cfg(self.cfg)

    def test_published_stats_without_dataset(self):
        dims, preprocessor = make_datasets(self.cfg.model_kwargs.data, self.cfg.model_kwargs.data_aug)
        self.assertTrue(dims.normalization_stats_only)
        self.assertEqual((dims.action_dim, dims.proprio_dim), (4, 4))
        action = torch.tensor([-1.0, 0.25, 0.5, 1.0])
        torch.testing.assert_close(preprocessor.denormalize_actions(preprocessor.normalize_actions(action)), action)

    @unittest.skipUnless(torch.cuda.is_available(), "CEM currently requires CUDA")
    def test_all_scored_and_captured_actions_respect_raw_bounds(self):
        _, preprocessor = make_datasets(self.cfg.model_kwargs.data, self.cfg.model_kwargs.data_aug)
        seen = []

        def unroll(z_init, act_suffix):
            raw = preprocessor.denormalize_actions(act_suffix.cpu().reshape(-1, 4))
            self.assertTrue((raw.abs() <= 1 + 1e-6).all())
            seen.append(raw)
            return act_suffix

        # The asymmetric normalized bounds must apply to all five packed frames.
        repeats = 5
        low = preprocessor.normalize_actions(-torch.ones(4)).repeat(repeats)
        high = preprocessor.normalize_actions(torch.ones(4)).repeat(repeats)
        self.assertFalse(torch.allclose(low, -torch.ones(20)))
        planner = CEMPlanner(
            unroll, iterations=3, num_samples=16, num_elites=4, horizon=2,
            action_dim=20, var_scale=10, num_act_stepped=1,
            capture_rounds=[1, 2, 3], capture_samples=True,
            action_lower_bounds=low, action_upper_bounds=high,
            local_generator=torch.Generator(device="cuda").manual_seed(7),
        )
        planner.set_objective(lambda predicted, actions: predicted.square().mean(dim=(0, 2)))
        result = planner.plan(torch.zeros(1, device="cuda"))
        self.assertGreater(len(seen), 3)
        self.assertEqual(len(result.info["diagnostic_candidates"]), 3 * 17)
        for candidate in result.info["diagnostic_candidates"]:
            raw = preprocessor.denormalize_actions(candidate["actions_normalized"].reshape(-1, 4))
            self.assertTrue((raw.abs() <= 1 + 1e-6).all())
        raw_returned = preprocessor.denormalize_actions(result.actions.cpu().reshape(-1, 4))
        self.assertTrue((raw_returned.abs() <= 1 + 1e-6).all())

    def test_snapshot_reproduces_mid_episode_trajectory_and_image(self):
        env = make_env(self.cfg)
        try:
            env.reset(seed=4)
            for _ in range(7):
                env.step(torch.tensor([0.1, 0.4, 0.2, 0.0]))
            snapshot = env.get_replay_state()
            elapsed = env.elapsed_steps()
            image = env.render().copy()
            actions = np.random.default_rng(12).uniform(-0.6, 0.6, size=(12, 4))

            def execute():
                trajectory = []
                for action in actions:
                    _, _, _, _, info = env.step(torch.tensor(action, dtype=torch.float32))
                    trajectory.append(info["state"].copy())
                return np.stack(trajectory)

            expected = execute()
            for _ in range(2):
                env.set_replay_state(snapshot)
                env.set_elapsed_steps(elapsed)
                np.testing.assert_array_equal(env.render(), image)
                np.testing.assert_array_equal(execute(), expected)
            # B/C will use a separately constructed simulator, not the original.
            env.close()
            env = make_env(self.cfg)
            env.reset(seed=91)
            env.set_replay_state(snapshot)
            env.set_elapsed_steps(elapsed)
            np.testing.assert_array_equal(env.render(), image)
            np.testing.assert_array_equal(execute(), expected)
            with self.assertRaisesRegex(ValueError, "outside the benchmark bounds"):
                env.step(torch.tensor([1.5, 0.0, 0.0, 0.0]))
        finally:
            env.close()

    def test_native_success_stops_an_action_batch(self):
        from metaworld.policies import SawyerReachWallV3Policy

        env = make_env(self.cfg)
        try:
            obs, info = env.reset(seed=4)
            policy = SawyerReachWallV3Policy()
            first_success = None
            for step in range(1, 151):
                action = np.clip(policy.get_action(info["state"]), -1, 1)
                _, _, done, _, info = env.step(torch.as_tensor(action, dtype=torch.float32))
                if info["success"]:
                    self.assertTrue(done)
                    first_success = step
                    break
            self.assertIsNotNone(first_success)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()

"""Resume must retain completed trials, RNG continuity, and final aggregation."""

import copy
from pathlib import Path
import random
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf
import torch

from evals.simu_env_planning import eval as evaluation
from evals.simu_env_planning.episode_resume import (
    archive_incomplete_episode, check_resume_config, config_digest,
    load_episode_checkpoint,
)


class FakeEnv:
    def __init__(self):
        self.value = 0
        self._frames, self._proprios = [], []

    def close(self):
        pass

    def get_replay_state(self):
        return self.value

    def set_replay_state(self, value):
        self.value = value

    def elapsed_steps(self):
        return self.value

    def set_elapsed_steps(self, value):
        self.value = value


class FakeAgent:
    def __init__(self, cfg, *args, **kwargs):
        self.local_generator = torch.Generator().manual_seed(cfg.local_seed)
        self.local_gpu_generator = torch.Generator().manual_seed(cfg.local_seed + 1)


class FakeEvaluator:
    fail_on = None
    calls = []

    def __init__(self, cfg, agent):
        pass

    def eval(self, cfg, agent, env, task_idx, ep):
        self.calls.append(ep)
        if ep == self.fail_on:
            raise RuntimeError("simulated interruption")
        # Exercise both agent streams, global streams, and environment continuity.
        value = (torch.rand(1, generator=agent.local_generator).item()
                 + torch.randn(7, generator=agent.local_gpu_generator).sum().item()
                 + torch.rand(1).item() + random.random() + np.random.random() + env.value)
        env.value += 1
        return (1., float(value > 0), value, 0., abs(value), -1., -1., -1., abs(value), -1., -1.)


class ResumeTests(unittest.TestCase):
    def config(self, path):
        raw = dict(work_dir=str(path), frameskip=5, resume_config_digest="test-digest",
                   meta=dict(seed=1, eval_episodes=2, quick_debug=False),
                   distributed=dict(distribute_multitask_eval=False),
                   task_specification=dict(task="mw-reach-wall", goal_source="expert"),
                   planner=dict(planner_name="cem", num_elites=2, num_samples=4, repeat_actskip=False))
        return raw

    def run_eval(self, cfg):
        with patch.object(evaluation, "make_env", side_effect=lambda cfg: FakeEnv()), \
             patch.object(evaluation, "GC_Agent", FakeAgent), \
             patch.object(evaluation, "PlanEvaluator", FakeEvaluator), \
             patch.object(evaluation, "Logger"), \
             patch.object(evaluation.dist, "get_world_size", return_value=1), \
             patch.object(evaluation.dist, "barrier"):
            evaluation.main_distributed_episodes_eval(cfg, model=SimpleNamespace(action_skip=1))

    def test_interrupted_resume_matches_uninterrupted_metrics_and_rng(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            FakeEvaluator.fail_on, FakeEvaluator.calls = None, []
            self.run_eval(self.config(root / "reference"))
            cfg = self.config(root / "interrupted")
            FakeEvaluator.fail_on = 1
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                self.run_eval(cfg)
            first = root / "interrupted/mw-reach-wall/ep_0/episode_result.pt"
            before = first.read_bytes()
            FakeEvaluator.fail_on, FakeEvaluator.calls = None, []
            cfg["meta"]["resume_completed"] = True
            self.run_eval(cfg)
            self.assertEqual(FakeEvaluator.calls, [1])
            self.assertEqual(first.read_bytes(), before)
            for ep in range(2):
                ref = load_episode_checkpoint(root / f"reference/mw-reach-wall/ep_{ep}", "test-digest")
                resumed = load_episode_checkpoint(root / f"interrupted/mw-reach-wall/ep_{ep}", "test-digest")
                self.assertEqual(ref["values"], resumed["values"])
                self.assertTrue(torch.equal(ref["rng"]["agent_gpu"], resumed["rng"]["agent_gpu"]))

    def test_config_changes_are_rejected_but_resume_flag_is_allowed(self):
        cfg = self.config(".")
        resumed = copy.deepcopy(cfg)
        resumed["meta"]["resume_completed"] = True
        check_resume_config(resumed, cfg)
        self.assertEqual(config_digest(OmegaConf.create(resumed)), config_digest(cfg))
        resumed["planner"]["num_samples"] += 1
        with self.assertRaisesRegex(ValueError, "differs"):
            check_resume_config(resumed, cfg)

    def test_execution_batching_and_monitoring_preserve_existing_checkpoint_digest(self):
        cfg = self.config(".")
        cfg["logging"] = {"save_csv": True}
        resumed = copy.deepcopy(cfg)
        resumed["planner"]["cost_batch_size"] = 32
        resumed["logging"]["resource_monitor_interval_s"] = 5
        check_resume_config(resumed, cfg)
        self.assertEqual(config_digest(resumed), config_digest(cfg))

    def test_uncommitted_checkpoint_is_not_completed_and_partial_is_archived(self):
        with TemporaryDirectory() as temp:
            ep = Path(temp) / "ep_1"
            ep.mkdir()
            (ep / "episode_result.pt.tmp").write_bytes(b"interrupted save")
            self.assertIsNone(load_episode_checkpoint(ep, "digest"))
            archived = archive_incomplete_episode(ep)
            self.assertFalse(ep.exists())
            self.assertEqual((archived / "episode_result.pt.tmp").read_bytes(), b"interrupted save")


if __name__ == "__main__":
    unittest.main()

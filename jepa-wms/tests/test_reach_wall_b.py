"""Checks for decision-local simulator replay, B scoring, and safe continuation."""

import csv
from pathlib import Path
import random
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from omegaconf import OmegaConf
import torch
from tensordict import TensorDict

from evals.simu_env_planning.envs.init import make_env
from evals.simu_env_planning.planning.common.parser import parse_cfg
from evals.simu_env_planning.planning.planning.objectives import ReprTargetDistMPCObjective
from evals.simu_env_planning.planning.reach_wall_b import (
    ScoreStore, encode_actual_endpoints, rollout_candidate, validate_start_image,
)

CONFIG = Path(__file__).resolve().parents[1] / "configs/evals/simu_env_planning/mw/jepa-wm/reach_wall_a.yaml"


def capture(env, obs, info):
    return dict(initial_state=torch.as_tensor(info["state"]).clone(),
                simulator_snapshot=env.get_replay_state(), episode_steps_elapsed=env.elapsed_steps(),
                episode_steps_left=env.steps_left(), initial_observation=dict(visual=obs.clone(), proprio=info["proprio"].clone()),
                pixel_frames=list(env._frames), proprio_frames=list(env._proprios),
                action_low=env.action_space.low, action_high=env.action_space.high,
                returned_actions_normalized=torch.zeros(3,20),
                rng=dict(python=random.getstate(), numpy=np.random.get_state(), torch_cpu=torch.get_rng_state()))


class ReachWallBTests(unittest.TestCase):
    def test_sparse_render_outlier_is_distinguished_from_changed_scene(self):
        image = torch.zeros(1,3,224,224)
        sparse = image.clone()
        sparse[0,1,175,9] = 16
        self.assertEqual(validate_start_image(sparse,image),16)
        with self.assertRaisesRegex(ValueError,"scene differs"):
            validate_start_image(image+2,image)

    def make_env(self):
        cfg = OmegaConf.load(CONFIG)
        cfg.work_dir, cfg.frameskip = ".", 5
        cfg.task_specification.done_at_succ = False
        return make_env(parse_cfg(cfg))

    def test_endpoint_only_render_matches_full_render_and_repeated_restoration(self):
        env = self.make_env()
        try:
            obs, info = env.reset(seed=4)
            bundle = capture(env, obs, info)
            candidate = dict(actions_simulator=torch.tensor(np.random.default_rng(15).uniform(-.5,.5,(20,4)), dtype=torch.float32),
                             actions_normalized=torch.zeros(4,20))
            fast, metrics, trace = rollout_candidate(env, bundle, candidate)
            full, other_metrics, other_trace = rollout_candidate(env, bundle, candidate, render_each_step=True)
            self.assertEqual(metrics, other_metrics)
            self.assertEqual(trace, other_trace)
            torch.testing.assert_close(fast["visual"], full["visual"], atol=1, rtol=0)
            torch.testing.assert_close(fast["proprio"], full["proprio"], atol=0, rtol=0)
            _, repeated_metrics, repeated_trace = rollout_candidate(env, bundle, candidate)
            self.assertEqual(metrics, repeated_metrics)
            self.assertEqual(trace, repeated_trace)
        finally:
            env.close()

    def test_native_success_is_recorded_while_diagnostic_horizon_continues(self):
        from metaworld.policies import SawyerReachWallV3Policy

        env = self.make_env()
        try:
            obs, info = env.reset(seed=4)
            bundle = capture(env, obs, info)
            actions = []
            policy = SawyerReachWallV3Policy()
            first = None
            for step in range(1,81):
                action = torch.as_tensor(np.clip(policy.get_action(info["state"]), -1, 1), dtype=torch.float32)
                actions.append(action)
                _, _, _, _, info = env.step(action)
                if info["success"] and first is None:
                    first = step
            self.assertIsNotNone(first)
            self.assertLess(first,80)
            candidate = dict(actions_simulator=torch.stack(actions),actions_normalized=torch.zeros(16,20))
            _, metrics, trace = rollout_candidate(env,bundle,candidate)
            self.assertEqual(metrics["task_first_hit_step"],first)
            self.assertEqual(metrics["task_steps"],first)
            self.assertEqual(metrics["full_horizon_steps"],80)
            self.assertEqual(len(trace),80)
        finally:
            env.close()

    def test_endpoint_batch_time_axes_and_original_objective(self):
        class Encoder:
            def encode(self, inputs):
                return TensorDict(inputs,batch_size=[])

        goal = TensorDict(dict(visual=torch.zeros(1,1,3,2,2),proprio=torch.zeros(1,1,4)),batch_size=[])
        objective = ReprTargetDistMPCObjective(None,goal,alpha=.1,sum_all_diffs=False)
        endpoints = [dict(visual=torch.full((1,3,2,2),value),proprio=torch.full((1,4),value)) for value in (1.,2.,3.)]
        costs = encode_actual_endpoints(Encoder(),objective,endpoints)
        np.testing.assert_allclose(costs,[1.1,4.4,9.9],atol=1e-6,rtol=0)

    def test_committed_scores_survive_resume_and_source_changes_are_rejected(self):
        with TemporaryDirectory() as temp:
            path = Path(temp)/"scores.sqlite3"
            store = ScoreStore(path,{"version":1})
            store.ensure_pool(0,0,"original",3)
            first = dict(episode=0,plan_call=0,candidate_index=0,a_cost=.3,b_cost=.2,is_deployed_output=False)
            last = dict(episode=0,plan_call=0,candidate_index=2,a_cost=.1,b_cost=.4,is_deployed_output=False)
            store.commit([first,last])
            store.connection.close()
            store = ScoreStore(path,{"version":1})
            self.assertEqual(set(store.rows(0,0)),{0,2})
            store.commit([dict(episode=0,plan_call=0,candidate_index=1,a_cost=.2,b_cost=.3,is_deployed_output=True)])
            store.export(Path(temp)/"scores.csv")
            with (Path(temp)/"scores.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([int(row["candidate_index"]) for row in rows],[0,1,2])
            store.export_selections(Path(temp)/"selections.csv")
            with (Path(temp)/"selections.csv").open(newline="") as handle:
                choices = {row["selection"]:int(row["candidate_index"]) for row in csv.DictReader(handle)}
            self.assertEqual(choices,dict(a_deployed_mean=1,a_pool_min=2,b_pool_min=0))
            with self.assertRaisesRegex(ValueError,"changed"):
                store.ensure_pool(0,0,"modified",3)
            store.connection.close()


if __name__ == "__main__":
    unittest.main()

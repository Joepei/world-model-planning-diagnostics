"""Contracts for oracle-scored CEM at B's own evolving simulator states."""

import copy
from pathlib import Path
import random
import unittest

import numpy as np
from omegaconf import OmegaConf
import torch
from tensordict import TensorDict

from evals.simu_env_planning.envs.init import make_env
from evals.simu_env_planning.planning.common.parser import parse_cfg
from evals.simu_env_planning.planning.planning.objectives import ReprTargetDistMPCObjective
from evals.simu_env_planning.planning.reach_wall_b import encode_actual_endpoints, rollout_candidate
from evals.simu_env_planning.planning.reach_wall_b_closed_loop import (
    OracleEndpointUnroll, build_planner, decision_snapshot, global_rng, raw_actions,
    validate_reference_costs,
)
from evals.utils import make_datasets

CONFIG = Path(__file__).resolve().parents[1] / "configs/evals/simu_env_planning/mw/jepa-wm/reach_wall_a.yaml"


class EncoderOnly:
    action_skip = 1

    def __init__(self, device="cpu"):
        self.device = device

    def encode(self, inputs):
        return TensorDict(dict(visual=inputs["visual"].to(self.device).float()/255,
                               proprio=inputs["proprio"].to(self.device).float()), batch_size=[])

    def unroll(self, *args, **kwargs):
        raise AssertionError("B must never call learned future prediction")


class ClosedLoopBTests(unittest.TestCase):
    def setUp(self):
        self.cfg = OmegaConf.load(CONFIG)
        self.cfg.work_dir, self.cfg.frameskip = ".", 5
        self.cfg = parse_cfg(self.cfg)
        _, self.preprocessor = make_datasets(self.cfg.model_kwargs.data, self.cfg.model_kwargs.data_aug)
        self.actor = make_env(self.cfg)
        diagnostic = copy.deepcopy(self.cfg)
        diagnostic.task_specification.done_at_succ = False
        self.scratch = make_env(diagnostic)
        self.visual, _ = self.actor.reset(seed=4)
        self.scorer = OracleEndpointUnroll(self.scratch, EncoderOnly(), self.preprocessor, batch_size=32)

    def tearDown(self):
        self.actor.close()
        self.scratch.close()

    def sequences(self):
        raw = torch.tensor(np.random.default_rng(33).uniform(-.4,.4,(3,10,4)),dtype=torch.float32)
        packed = self.preprocessor.normalize_actions(raw).reshape(3,2,20).transpose(0,1)
        return raw, packed

    def test_packing_roundtrip_keeps_time_frame_and_component_order(self):
        raw, packed = self.sequences()
        for index in range(3):
            torch.testing.assert_close(raw_actions(packed[:,index],self.preprocessor),raw[index],atol=1e-7,rtol=0)

    def test_oracle_cost_matches_offline_b_and_cannot_move_actor_or_rng(self):
        bundle = decision_snapshot(self.actor,self.visual,self.cfg)
        self.scorer.set_decision(bundle)
        _, packed = self.sequences()
        goal = TensorDict(dict(visual=torch.zeros(1,1,3,224,224),proprio=torch.zeros(1,1,4)),batch_size=[])
        objective = ReprTargetDistMPCObjective(None,goal,alpha=.1)
        before = global_rng()
        actor_state = self.actor.get_replay_state()
        encoded = self.scorer(None,packed)
        after = global_rng()
        self.assertEqual(before["python"],after["python"])
        self.assertEqual(before["numpy"][0],after["numpy"][0])
        np.testing.assert_array_equal(before["numpy"][1],after["numpy"][1])
        self.assertEqual(before["numpy"][2:],after["numpy"][2:])
        self.assertTrue(torch.equal(before["torch_cpu"],after["torch_cpu"]))
        for first,last in zip(before["cuda"],after["cuda"]):self.assertTrue(torch.equal(first,last))
        np.testing.assert_array_equal(actor_state["integration_state"],self.actor.get_replay_state()["integration_state"])
        self.assertEqual(self.actor.elapsed_steps(),0)
        endpoints=[]
        for sequence in packed.transpose(0,1):
            endpoint,_,_=rollout_candidate(self.scratch,bundle,dict(actions_normalized=sequence,
                actions_simulator=raw_actions(sequence,self.preprocessor)))
            endpoints.append(endpoint)
        expected=encode_actual_endpoints(EncoderOnly(),objective,endpoints)
        np.testing.assert_allclose(objective(encoded,None).numpy(),expected,atol=1e-7,rtol=0)

    def test_next_decision_uses_new_actor_state(self):
        _,packed=self.sequences()
        first=decision_snapshot(self.actor,self.visual,self.cfg)
        self.scorer.set_decision(first)
        initial=self.scorer(None,packed)["proprio"].clone()
        for _ in range(5):
            self.visual,_,_,_,_=self.actor.step(torch.tensor([.7,.5,.2,0.]))
        second=decision_snapshot(self.actor,self.visual,self.cfg)
        self.scorer.set_decision(second)
        later=self.scorer(None,packed)["proprio"]
        self.assertFalse(torch.equal(initial,later))
        self.assertEqual(second["episode_steps_elapsed"],5)
        self.assertEqual(self.actor.elapsed_steps(),5)

    @unittest.skipUnless(torch.cuda.is_available(),"Existing CEM uses CUDA")
    def test_cem_updates_using_oracle_cost_and_returns_final_mean_prefix(self):
        cfg=copy.deepcopy(self.cfg)
        cfg.planner.iterations=2;cfg.planner.num_samples=8;cfg.planner.num_elites=3
        cfg.planner.horizon=2;cfg.planner.num_act_stepped=1;cfg.planner.capture_rounds=[1,2]
        self.scorer.model=EncoderOnly("cuda")
        self.scorer.set_decision(decision_snapshot(self.actor,self.visual,cfg))
        planner=build_planner(cfg,self.scorer,self.preprocessor,torch.Generator(device="cuda").manual_seed(2))
        goal=TensorDict(dict(visual=torch.zeros(1,1,3,224,224,device="cuda"),
                            proprio=torch.zeros(1,1,4,device="cuda")),batch_size=[])
        planner.set_objective(ReprTargetDistMPCObjective(None,goal,alpha=.1))
        result=planner.plan(torch.zeros(1,device="cuda"))
        candidates=result.info["diagnostic_candidates"]
        self.assertEqual(len(candidates),18)
        final=next(row for row in candidates if row["is_deployed_output"])
        torch.testing.assert_close(result.actions.cpu(),final["actions_normalized"][:1],rtol=0,atol=0)
        for candidate in candidates:
            self.assertTrue((raw_actions(candidate["actions_normalized"],self.preprocessor).abs()<=1+1e-6).all())
        self.assertEqual(self.actor.elapsed_steps(),0)


class ReferenceCostValidationTests(unittest.TestCase):
    def test_measured_cross_run_drift_is_bounded_but_wrong_costs_are_rejected(self):
        saved = [2.3301596641540527, 2.451181173324585, 2.3646490573883057, 2.315178394317627]
        replay = [2.330148696899414, 2.4516842365264893, 2.364650011062622, 2.3151779174804688]
        self.assertLess(validate_reference_costs(replay, saved), .00051)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            validate_reference_costs(torch.tensor(saved) * 1.01, saved)

    def test_invalid_reference_values_are_rejected(self):
        for actual in ([float("nan")], [float("inf")], []):
            with self.subTest(actual=actual), self.assertRaises(ValueError):
                validate_reference_costs(actual, [1.])


if __name__=="__main__":
    unittest.main()

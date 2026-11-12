"""C's exact task ordering, oracle isolation, early stopping, and CEM contract."""

import copy
from pathlib import Path
import unittest

import numpy as np
from omegaconf import OmegaConf
import torch
from tensordict import TensorDict

from evals.simu_env_planning.envs.init import make_env
from evals.simu_env_planning.planning.common.parser import parse_cfg
from evals.simu_env_planning.planning.reach_wall_b import restore_decision_state, rollout_candidate
from evals.simu_env_planning.planning.reach_wall_b_closed_loop import build_planner, decision_snapshot, global_rng, raw_actions
from evals.simu_env_planning.planning.reach_wall_c_closed_loop import OracleTaskUnroll, TaskObjective, task_rollout
from evals.utils import make_datasets

CONFIG = Path(__file__).resolve().parents[1] / "configs/evals/simu_env_planning/mw/jepa-wm/reach_wall_a.yaml"


def outcomes(success,hits,distances):
    return TensorDict(dict(success=torch.tensor([success]),first_hit_step=torch.tensor([hits]),
        distance_m=torch.tensor([distances],dtype=torch.float64),horizon_raw_actions=torch.full((1,len(hits)),30)),batch_size=[])


class TaskOrderingTests(unittest.TestCase):
    def test_success_priority_arrival_ties_and_failure_metres(self):
        values=outcomes([True,True,True,False,False],[30,2,2,-1,-1],[.01,.02,.04,.051,2.])
        key=TaskObjective()(values)
        self.assertLess(key[1],key[0])
        self.assertEqual(key[1],key[2])
        self.assertLess(key[0],key[3])
        self.assertLess(key[3],key[4])

    def test_full_population_and_microbatches_have_identical_order_keys(self):
        rng=np.random.default_rng(72)
        success=rng.random(300)<.5
        hits=np.where(success,rng.integers(0,31,300),-1)
        distances=np.where(success,rng.uniform(.001,.05,300),rng.uniform(.051,2,300))
        values=outcomes(success.tolist(),hits.tolist(),distances.tolist())
        full=TaskObjective()(values)
        batched=torch.cat([TaskObjective()(TensorDict({key:value[:,start:start+32] for key,value in values.items()},batch_size=[]))
                           for start in range(0,300,32)])
        torch.testing.assert_close(full,batched,rtol=0,atol=0)
        expected=sorted(range(300),key=lambda i:(0,int(hits[i])) if success[i] else (1,float(distances[i])))
        self.assertEqual(sorted(range(300),key=lambda i:float(full[i])),expected)

    def test_inconsistent_native_outcomes_are_rejected(self):
        for value in (outcomes([True],[31],[.01]),outcomes([False],[-1],[.01]),outcomes([True],[2],[float("nan")])):
            with self.assertRaises(ValueError):TaskObjective()(value)


class CControllerTests(unittest.TestCase):
    def setUp(self):
        self.cfg=OmegaConf.load(CONFIG)
        self.cfg.work_dir,self.cfg.frameskip=".",5
        self.cfg=parse_cfg(self.cfg)
        _,self.preprocessor=make_datasets(self.cfg.model_kwargs.data,self.cfg.model_kwargs.data_aug)
        self.actor=make_env(self.cfg)
        diagnostic=copy.deepcopy(self.cfg)
        diagnostic.task_specification.done_at_succ=False
        self.scratch=make_env(diagnostic)
        self.visual,_=self.actor.reset(seed=4)

    def tearDown(self):
        self.actor.close();self.scratch.close()

    def test_early_stop_agrees_with_full_native_reference(self):
        from metaworld.policies import SawyerReachWallV3Policy
        bundle=decision_snapshot(self.actor,self.visual,self.cfg)
        restore_decision_state(self.scratch,bundle)
        policy=SawyerReachWallV3Policy();actions=[]
        state=bundle["initial_state"].numpy()
        for _ in range(80):
            action=torch.as_tensor(np.clip(policy.get_action(state),-1,1),dtype=torch.float32)
            actions.append(action)
            _,_,_,_,info=self.scratch.step(action)
            state=np.asarray(info["state"])
        raw=torch.stack(actions)
        packed=self.preprocessor.normalize_actions(raw).reshape(16,20)
        metrics,trace=task_rollout(self.scratch,bundle,packed,self.preprocessor)
        _,reference,full_trace=rollout_candidate(self.scratch,bundle,dict(actions_normalized=packed,actions_simulator=raw_actions(packed,self.preprocessor)))
        self.assertTrue(metrics["task_success"])
        self.assertEqual(metrics["task_first_hit_step"],reference["task_first_hit_step"])
        self.assertEqual(metrics["task_end_distance_m"],reference["task_end_distance_m"])
        self.assertEqual(trace,full_trace[:metrics["task_first_hit_step"]])
        self.assertLess(metrics["simulator_actions"],80)
        self.assertEqual(self.actor.elapsed_steps(),0)

    def test_scorer_preserves_actor_and_rng_then_uses_new_decision(self):
        oracle=OracleTaskUnroll(self.scratch,self.preprocessor)
        bundle=decision_snapshot(self.actor,self.visual,self.cfg)
        oracle.set_decision(bundle)
        raw=torch.zeros(10,4)
        packed=self.preprocessor.normalize_actions(raw).reshape(2,20).unsqueeze(1)
        before=global_rng()
        first=oracle(None,packed)
        after=global_rng()
        self.assertEqual(before["python"],after["python"])
        np.testing.assert_array_equal(before["numpy"][1],after["numpy"][1])
        self.assertTrue(torch.equal(before["torch_cpu"],after["torch_cpu"]))
        for a,b in zip(before["cuda"],after["cuda"]):self.assertTrue(torch.equal(a,b))
        np.testing.assert_array_equal(bundle["simulator_snapshot"]["integration_state"],self.actor.get_replay_state()["integration_state"])
        for _ in range(5):self.visual,_,_,_,_=self.actor.step(torch.tensor([.7,.5,.2,0.]))
        oracle.set_decision(decision_snapshot(self.actor,self.visual,self.cfg))
        second=oracle(None,packed)
        self.assertFalse(torch.equal(first["distance_m"],second["distance_m"]))
        self.assertEqual(self.actor.elapsed_steps(),5)

    @unittest.skipUnless(torch.cuda.is_available(),"Existing paired CEM uses CUDA")
    def test_cem_elites_follow_c_order_and_return_final_mean(self):
        cfg=copy.deepcopy(self.cfg)
        cfg.planner.iterations=2;cfg.planner.num_samples=8;cfg.planner.num_elites=3
        cfg.planner.horizon=2;cfg.planner.num_act_stepped=1;cfg.planner.capture_rounds=[1,2]
        oracle=OracleTaskUnroll(self.scratch,self.preprocessor)
        oracle.set_decision(decision_snapshot(self.actor,self.visual,cfg))
        planner=build_planner(cfg,oracle,self.preprocessor,torch.Generator(device="cuda").manual_seed(6))
        planner.set_objective(TaskObjective())
        result=planner.plan(torch.zeros(1,device="cuda"))
        rows=result.info["diagnostic_candidates"]
        self.assertEqual(len(rows),18)
        for round_index in (1,2):
            batch=[r for r in rows if r["cem_round"]==round_index and r["origin"]!="updated_mean"]
            self.assertLessEqual(max(r["a_cost"] for r in batch if r["is_elite"]),min(r["a_cost"] for r in batch if not r["is_elite"]))
        final=next(r for r in rows if r["is_deployed_output"])
        torch.testing.assert_close(result.actions.cpu(),final["actions_normalized"][:1],rtol=0,atol=0)
        self.assertEqual(self.actor.elapsed_steps(),0)


if __name__=="__main__":
    unittest.main()

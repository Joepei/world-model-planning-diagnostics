"""C ordering, horizon semantics, and state-local selection/regret contracts."""

import unittest

from evals.simu_env_planning.planning.reach_wall_c import c_outcome,analyze_pool,aggregate_results


def candidate(index,success=False,hit=None,distance=.1,a=1.,b=1.,**extra):
    prefix_success=success and hit<=15
    row=dict(episode=0,plan_call=0,candidate_index=index,candidate=f"c{index}",route_id=f"route{index}",
             a_cost=a,b_cost=b,cem_round=1,origin="updated_mean" if index==3 else "included_mean" if index==0 else "gaussian_sample",
             sample_index=None if index==3 else index,is_elite=True if index==1 else None if index==3 else False,
             is_deployed_output=index==3,task_success=success,task_first_hit_step=hit,
             task_steps=hit if success else 30,task_end_distance_m=distance,
             full_horizon_steps=30,full_horizon_goal_distance_m=.1,
             prefix_steps=15,prefix_success=prefix_success,prefix_task_steps=hit if prefix_success else 15,
             prefix_task_end_distance_m=distance if prefix_success else .09,
             episode_steps_elapsed=0,episode_steps_left=100)
    row.update(extra)
    return row


def pool():
    return [candidate(0,a=.1,b=2.),candidate(1,True,20,.04,a=.8,b=.7),
            candidate(2,True,10,.03,a=1.2,b=.6),candidate(3,a=.9,b=1.)]


class ReachWallCTests(unittest.TestCase):
    def test_full_horizon_success_does_not_imply_prefix_success(self):
        row=candidate(1,True,20,.04)
        self.assertTrue(c_outcome(row,"full_horizon")["c_success"])
        self.assertFalse(c_outcome(row,"executed_prefix")["c_success"])
        self.assertIsNone(c_outcome(row,"executed_prefix")["c_first_hit_step"])

    def test_first_hit_success_is_retained_even_if_diagnostic_endpoint_is_outside(self):
        row=candidate(1,True,20,.04)
        self.assertTrue(c_outcome(row)["c_success"])
        row["task_steps"]=21
        with self.assertRaisesRegex(ValueError,"First-hit"):
            c_outcome(row)

    def test_deployed_mean_is_distinct_from_pool_minimum_and_regret_is_conditional(self):
        _,summary,choices,_=analyze_pool(pool(),"full_horizon")
        choices={row["selection"]:row for row in choices}
        self.assertEqual(choices["a_deployed_mean"]["candidate"],"c3")
        self.assertEqual(choices["a_pool_min"]["candidate"],"c0")
        self.assertEqual(choices["b_pool_min"]["candidate"],"c2")
        self.assertEqual(choices["a_deployed_mean"]["c_missed_success"],1)
        self.assertIsNone(choices["a_deployed_mean"]["c_excess_distance_m"])
        self.assertIsNone(choices["a_deployed_mean"]["c_excess_steps"])
        self.assertEqual(summary["b_vs_a_deployed_mean"],"better")

    def test_equal_first_hit_successes_tie_without_distance_tiebreaker(self):
        rows=pool()
        rows[1]=candidate(1,True,10,.049,a=.8,b=.7)
        ranked,_,_,_=analyze_pool(rows,"full_horizon")
        lookup={row["candidate"]:row for row in ranked}
        self.assertEqual(lookup["c1"]["c_rank"],lookup["c2"]["c_rank"])

    def test_failed_pool_regret_is_in_metres(self):
        rows=[candidate(i,distance=.06+.02*i,a=4-i,b=i+1) for i in range(4)]
        _,_,choices,_=analyze_pool(rows,"full_horizon")
        deployed=next(row for row in choices if row["selection"]=="a_deployed_mean")
        self.assertAlmostEqual(deployed["c_excess_distance_m"],.06)
        self.assertIsNone(deployed["c_excess_steps"])
        self.assertEqual(deployed["c_missed_success"],0)

    def test_different_planning_calls_cannot_be_pooled(self):
        rows=pool();rows[1]["plan_call"]=1
        with self.assertRaisesRegex(ValueError,"mix decision states"):
            analyze_pool(rows,"full_horizon")

    def test_aggregate_reports_coverage_and_separate_views(self):
        summaries,choices=[],[]
        for scope in ("full_horizon","executed_prefix"):
            _,summary,selected,_=analyze_pool(pool(),scope)
            summaries.append(summary);choices.extend(selected)
        result=aggregate_results(summaries,choices,3)
        self.assertFalse(result["complete"])
        self.assertEqual(result["completed_pools"],1)
        self.assertEqual(result["scopes"]["full_horizon"]["selections"]["b_pool_min"]["successes"],1)


if __name__=="__main__":
    unittest.main()

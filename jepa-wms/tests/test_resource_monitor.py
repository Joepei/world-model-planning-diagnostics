"""Telemetry survives missing GPU tools and appends flushed, readable samples."""

import csv
import copy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from evals.simu_env_planning.resource_monitor import ResourceMonitor


class ResourceMonitorTests(unittest.TestCase):
    def test_flushed_samples_append_across_sessions(self):
        fake_gpu = dict(gpu_memory_used_mib=1024., gpu_memory_total_mib=16376., gpu_temperature_c=45.)
        with TemporaryDirectory() as temp, patch("evals.simu_env_planning.resource_monitor.gpu_stats", return_value=fake_gpu):
            for _ in range(2):
                with ResourceMonitor(temp, interval_s=.01) as monitor:
                    monitor.set_context("planning", episode=6, plan_call=0)
            path = Path(temp) / "resource_usage.csv"
            with path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertGreaterEqual(len(rows), 2)
            self.assertEqual(path.read_text().count("timestamp_utc"), 1)
            self.assertTrue(all(float(row["system_ram_total_mib"]) > 0 for row in rows))
            self.assertTrue(all(float(row["process_rss_mib"]) > 0 for row in rows))
            self.assertTrue(all(row["gpu_temperature_c"] == "45.0" for row in rows))
            self.assertFalse(monitor.thread.is_alive())

    def test_missing_gpu_tool_preserves_ram_logging(self):
        with TemporaryDirectory() as temp, patch("evals.simu_env_planning.resource_monitor.gpu_stats", side_effect=FileNotFoundError("nvidia-smi")):
            sample = ResourceMonitor(temp, interval_s=5).sample()
            self.assertGreater(sample["system_ram_total_mib"], 0)
            self.assertIn("gpu:", sample["sampling_error"])

    def test_eval_entrypoint_records_session_and_passes_monitor(self):
        from evals.simu_env_planning import eval as evaluation

        with TemporaryDirectory() as temp:
            config = dict(folder=temp, tag="test", meta=dict(seed=1),
                          task_specification=dict(task="mw-reach-wall", goal_source="expert"),
                          logging=dict(resource_monitor_interval_s=.01), planner=dict(cost_batch_size=32),
                          model_kwargs=dict(module_name="math", checkpoint="unused", data=dict(custom=dict(frameskip=5))))
            observed = []

            def check_entrypoint(args, **kwargs):
                self.assertEqual(args["frameskip"], 5)
                self.assertIsInstance(kwargs["resource_monitor"], ResourceMonitor)
                observed.append(kwargs["resource_monitor"])

            with patch.object(evaluation, "init_distributed", return_value=(1, 0)), \
                 patch.object(evaluation, "make_datasets", return_value=(SimpleNamespace(action_dim=4, proprio_dim=4), None)), \
                 patch.object(evaluation, "init_module"), \
                 patch.object(evaluation, "main_distributed_episodes_eval", side_effect=check_entrypoint), \
                 patch("evals.simu_env_planning.resource_monitor.gpu_stats", return_value={}):
                evaluation.main(copy.deepcopy(config))
                original = Path(temp, "simu_env_planning/test/args_eval.yaml").read_bytes()
                resumed = copy.deepcopy(config)
                resumed["meta"]["resume_completed"] = True
                evaluation.main(resumed)
                self.assertEqual(Path(temp, "simu_env_planning/test/args_eval.yaml").read_bytes(), original)
            self.assertEqual(len(observed), 2)
            self.assertTrue(all(not monitor.thread.is_alive() for monitor in observed))
            sessions = Path(temp, "simu_env_planning/test/execution_sessions.jsonl").read_text().splitlines()
            self.assertEqual(len(sessions), 2)


if __name__ == "__main__":
    unittest.main()

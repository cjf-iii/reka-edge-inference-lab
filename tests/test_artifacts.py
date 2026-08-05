from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from reka_inference_lab.artifacts import verify_bundle
from reka_inference_lab.benchmark import simulate_benchmark
from reka_inference_lab.capacity import build_capacity_plan
from reka_inference_lab.models import Pricing, PromptCase, Workload
from reka_inference_lab.report import write_report_bundle


class ArtifactTests(unittest.TestCase):
    def _inputs(self):
        benchmark = simulate_benchmark(
            [PromptCase(case_id="p", prompt="test")],
            request_count=8,
            concurrency=2,
            model="demo",
        )
        workload = Workload.from_mapping(
            {
                "name": "artifact test",
                "modality": "text",
                "steady_requests_per_second": 0.5,
                "peak_multiplier": 2,
                "average_input_tokens": 10,
                "average_output_tokens": 20,
                "monthly_requests": 1000,
                "target_p95_ms": 3000,
                "target_utilization": 0.7,
            }
        )
        pricing = Pricing.from_mapping(
            {
                "name": "source",
                "checked_at": "2026-08-04",
                "source_url": "https://example.com/pricing",
                "input_per_million_tokens": 1,
                "output_per_million_tokens": 2,
            }
        )
        return benchmark, build_capacity_plan(benchmark, workload, pricing)

    def test_report_bundle_verifies(self) -> None:
        benchmark, plan = self._inputs()
        with tempfile.TemporaryDirectory() as temporary:
            write_report_bundle(temporary, benchmark, plan)
            result = verify_bundle(temporary)
            self.assertTrue(result["ok"])
            self.assertEqual(result["evidence_type"], "simulated")

    def test_verifier_rejects_mismatched_evidence(self) -> None:
        benchmark, plan = self._inputs()
        with tempfile.TemporaryDirectory() as temporary:
            write_report_bundle(temporary, benchmark, plan)
            plan_path = Path(temporary) / "capacity_plan.json"
            data = json.loads(plan_path.read_text())
            data["evidence_type"] = "observed-endpoint"
            plan_path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "does not match"):
                verify_bundle(temporary)


if __name__ == "__main__":
    unittest.main()

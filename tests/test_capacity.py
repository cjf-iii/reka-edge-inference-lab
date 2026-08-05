from __future__ import annotations

import unittest

from reka_inference_lab.benchmark import simulate_benchmark
from reka_inference_lab.capacity import build_capacity_plan
from reka_inference_lab.models import Pricing, PromptCase, Workload


class CapacityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.benchmark = simulate_benchmark(
            [PromptCase(case_id="p", prompt="test")],
            request_count=12,
            concurrency=3,
            model="demo",
        )
        self.workload = Workload.from_mapping(
            {
                "name": "buyer pilot",
                "modality": "video",
                "steady_requests_per_second": 1,
                "peak_multiplier": 3,
                "average_input_tokens": 100,
                "average_output_tokens": 50,
                "monthly_requests": 10000,
                "target_p95_ms": 2500,
                "target_utilization": 0.7,
                "video_minutes_indexed_per_month": 100,
                "video_searches_per_month": 1000,
                "privacy_requirement": "private",
            }
        )
        self.pricing = Pricing.from_mapping(
            {
                "name": "public units",
                "checked_at": "2026-08-04",
                "source_url": "https://example.com/pricing",
                "input_per_million_tokens": 1,
                "output_per_million_tokens": 2,
                "video_index_per_minute": 0.05,
                "video_search_each": 0.005,
            }
        )

    def test_plan_uses_request_and_token_constraints(self) -> None:
        plan = build_capacity_plan(self.benchmark, self.workload, self.pricing)
        estimate = plan["estimate"]
        self.assertEqual(
            estimate["estimated_capacity_units"],
            max(
                estimate["units_by_request_throughput"],
                estimate["units_by_output_token_throughput"],
            ),
        )
        self.assertGreaterEqual(estimate["estimated_capacity_units"], 1)

    def test_cost_components_are_traceable(self) -> None:
        plan = build_capacity_plan(self.benchmark, self.workload, self.pricing)
        cost = plan["monthly_public_unit_cost"]
        self.assertAlmostEqual(cost["input_tokens"], 1.0)
        self.assertAlmostEqual(cost["output_tokens"], 1.0)
        self.assertAlmostEqual(cost["video_indexing"], 5.0)
        self.assertAlmostEqual(cost["video_search"], 5.0)
        self.assertAlmostEqual(cost["total"], 12.0)

    def test_private_workload_generates_security_question(self) -> None:
        plan = build_capacity_plan(self.benchmark, self.workload, self.pricing)
        text = " ".join(plan["recommendations"])
        self.assertIn("retention", text)


if __name__ == "__main__":
    unittest.main()

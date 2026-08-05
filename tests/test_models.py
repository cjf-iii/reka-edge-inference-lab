from __future__ import annotations

import unittest

from reka_inference_lab.models import Pricing, PromptCase, Workload


class InputModelTests(unittest.TestCase):
    def test_workload_rejects_invalid_utilization(self) -> None:
        with self.assertRaisesRegex(ValueError, "below 1.0"):
            Workload.from_mapping(
                {
                    "name": "bad",
                    "modality": "text",
                    "steady_requests_per_second": 1,
                    "peak_multiplier": 2,
                    "average_input_tokens": 10,
                    "average_output_tokens": 10,
                    "monthly_requests": 100,
                    "target_p95_ms": 1000,
                    "target_utilization": 1,
                }
            )

    def test_pricing_preserves_dated_source(self) -> None:
        pricing = Pricing.from_mapping(
            {
                "name": "dated",
                "checked_at": "2026-08-04",
                "source_url": "https://example.com/pricing",
                "input_per_million_tokens": 1,
                "output_per_million_tokens": 2,
                "notes": ["estimate"],
            }
        )
        self.assertEqual(pricing.checked_at, "2026-08-04")
        self.assertEqual(pricing.notes, ("estimate",))

    def test_multimodal_prompt_requires_media_url(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires media_url"):
            PromptCase.from_mapping({"case_id": "image", "prompt": "describe", "modality": "image"})


if __name__ == "__main__":
    unittest.main()

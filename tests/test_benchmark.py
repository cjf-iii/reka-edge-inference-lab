from __future__ import annotations

import asyncio
import json
import unittest

import httpx

from reka_inference_lab.benchmark import (
    EndpointConfig,
    OpenAICompatibleBenchmark,
    SimulatedProfile,
    percentile,
    simulate_benchmark,
    simulate_concurrency_sweep,
)
from reka_inference_lab.models import PromptCase


class BenchmarkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.prompts = [PromptCase(case_id="p1", prompt="Explain inference")]

    def test_percentile_interpolates(self) -> None:
        self.assertEqual(percentile([1, 2, 3, 4], 50), 2.5)
        self.assertAlmostEqual(percentile([1, 2, 3, 4], 95), 3.85)

    def test_simulation_is_explicit_and_reconciles_failures(self) -> None:
        result = simulate_benchmark(
            self.prompts,
            request_count=10,
            concurrency=2,
            model="demo",
            profile=SimulatedProfile(failure_every=5),
        )
        self.assertEqual(result["evidence_type"], "simulated")
        self.assertEqual(result["summary"]["failure_count"], 2)
        self.assertEqual(result["summary"]["success_count"], 8)
        self.assertEqual(len(result["records"]), 10)

    def test_simulated_sweep_preserves_each_level(self) -> None:
        result = simulate_concurrency_sweep(
            self.prompts,
            request_count_per_level=4,
            concurrency_levels=[1, 2, 4],
            model="demo",
        )
        self.assertEqual(result["evidence_type"], "simulated")
        self.assertEqual(result["concurrency_levels"], [1, 2, 4])
        self.assertEqual([run["summary"]["concurrency"] for run in result["runs"]], [1, 2, 4])
        self.assertTrue(all(len(run["records"]) == 4 for run in result["runs"]))

    def test_openai_stream_records_ttft_and_provider_usage(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            self.assertTrue(payload["stream"])
            body = "\n\n".join(
                [
                    'data: {"choices":[{"delta":{"content":"hello "}}]}',
                    'data: {"choices":[{"delta":{"content":"world"}}]}',
                    'data: {"choices":[],"usage":{"prompt_tokens":7,"completion_tokens":2}}',
                    "data: [DONE]",
                    "",
                ]
            ).encode()
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)

        transport = httpx.MockTransport(handler)
        benchmark = OpenAICompatibleBenchmark(
            EndpointConfig(
                base_url="https://endpoint.example/v1",
                model="test-model",
                api_key="test-only",
            ),
            transport=transport,
        )
        result = asyncio.run(benchmark.run(self.prompts, request_count=2, concurrency=2))
        self.assertEqual(result["summary"]["success_count"], 2)
        self.assertEqual(result["records"][0]["output_text"], "hello world")
        self.assertEqual(result["records"][0]["output_tokens"], 2)
        self.assertEqual(result["records"][0]["token_count_source"], "provider")
        self.assertIsNotNone(result["records"][0]["ttft_ms"])

    def test_endpoint_failure_is_preserved_as_evidence(self) -> None:
        async def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, json={"error": "rate limited"})

        benchmark = OpenAICompatibleBenchmark(
            EndpointConfig(
                base_url="https://endpoint.example",
                model="test-model",
                api_key="test-only",
            ),
            transport=httpx.MockTransport(handler),
        )
        result = asyncio.run(benchmark.run(self.prompts, request_count=1, concurrency=1))
        self.assertEqual(result["summary"]["failure_count"], 1)
        self.assertIn("429", result["records"][0]["error"])

    def test_endpoint_sweep_preserves_excluded_warmup(self) -> None:
        async def handler(_request: httpx.Request) -> httpx.Response:
            body = b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n'
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)

        benchmark = OpenAICompatibleBenchmark(
            EndpointConfig(
                base_url="https://endpoint.example",
                model="test-model",
                api_key="test-only",
            ),
            transport=httpx.MockTransport(handler),
        )
        result = asyncio.run(
            benchmark.run_sweep(
                self.prompts,
                request_count_per_level=2,
                concurrency_levels=[1, 2],
                warmup_requests_per_level=1,
            )
        )
        self.assertEqual(result["concurrency_levels"], [1, 2])
        self.assertEqual(len(result["runs"]), 2)
        for run in result["runs"]:
            self.assertEqual(run["summary"]["request_count"], 2)
            self.assertTrue(run["warmup"]["excluded_from_measurement"])
            self.assertEqual(run["warmup"]["benchmark"]["summary"]["request_count"], 1)


if __name__ == "__main__":
    unittest.main()

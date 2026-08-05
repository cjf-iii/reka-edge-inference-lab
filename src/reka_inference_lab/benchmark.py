"""Inference benchmarking with raw evidence and explicit simulation boundaries."""

from __future__ import annotations

import asyncio
import json
import math
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

import httpx

from .models import PromptCase, RequestRecord


def percentile(values: Sequence[float], percentile_value: float) -> float:
    """Return a linearly interpolated percentile without a NumPy dependency."""

    if not values:
        raise ValueError("Cannot calculate a percentile of an empty sequence")
    if not 0 <= percentile_value <= 100:
        raise ValueError("percentile_value must be between 0 and 100")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile_value / 100
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _distribution(values: Iterable[float]) -> dict[str, float] | None:
    materialized = list(values)
    if not materialized:
        return None
    return {
        "mean": sum(materialized) / len(materialized),
        "p50": percentile(materialized, 50),
        "p95": percentile(materialized, 95),
        "p99": percentile(materialized, 99),
        "min": min(materialized),
        "max": max(materialized),
    }


def summarize_records(
    records: Sequence[RequestRecord], *, elapsed_seconds: float, concurrency: int
) -> dict[str, Any]:
    """Derive service metrics from raw request records."""

    if not records:
        raise ValueError("At least one request record is required")
    if elapsed_seconds <= 0:
        raise ValueError("elapsed_seconds must be positive")
    successful = [record for record in records if record.status == "ok"]
    failed = [record for record in records if record.status != "ok"]
    output_tokens = sum(record.output_tokens or 0 for record in successful)
    inter_token = [
        (record.e2e_ms - record.ttft_ms) / max((record.output_tokens or 1) - 1, 1)
        for record in successful
        if record.ttft_ms is not None and record.output_tokens
    ]
    return {
        "request_count": len(records),
        "success_count": len(successful),
        "failure_count": len(failed),
        "success_rate": len(successful) / len(records),
        "concurrency": concurrency,
        "elapsed_seconds": elapsed_seconds,
        "request_throughput_per_second": len(successful) / elapsed_seconds,
        "output_tokens_per_second": output_tokens / elapsed_seconds,
        "e2e_ms": _distribution(record.e2e_ms for record in successful),
        "ttft_ms": _distribution(
            record.ttft_ms for record in successful if record.ttft_ms is not None
        ),
        "inter_token_latency_ms": _distribution(inter_token),
        "token_count_sources": sorted(
            {record.token_count_source for record in successful if record.output_tokens is not None}
        ),
    }


@dataclass(frozen=True)
class SimulatedProfile:
    """Transparent synthetic endpoint parameters used only for the offline demo."""

    base_ttft_ms: float = 110.0
    output_tokens_per_second: float = 42.0
    queue_penalty_ms: float = 18.0
    failure_every: int = 0


def validate_concurrency_levels(levels: Sequence[int]) -> list[int]:
    """Return distinct positive concurrency levels in the requested order."""

    normalized = [int(level) for level in levels]
    if not normalized:
        raise ValueError("At least one concurrency level is required")
    if any(level <= 0 for level in normalized):
        raise ValueError("Concurrency levels must be positive")
    if len(set(normalized)) != len(normalized):
        raise ValueError("Concurrency levels must be distinct")
    return normalized


def simulate_benchmark(
    prompts: Sequence[PromptCase],
    *,
    request_count: int,
    concurrency: int,
    model: str,
    profile: SimulatedProfile | None = None,
) -> dict[str, Any]:
    """Create deterministic scheduling evidence without pretending it is measured inference."""

    if request_count <= 0 or concurrency <= 0:
        raise ValueError("request_count and concurrency must be positive")
    if not prompts:
        raise ValueError("At least one prompt is required")
    profile = profile or SimulatedProfile()
    lane_available = [0.0] * concurrency
    records: list[RequestRecord] = []
    for index in range(request_count):
        lane = min(range(concurrency), key=lane_available.__getitem__)
        started = lane_available[lane]
        case = prompts[index % len(prompts)]
        output_tokens = 36 + (index % 5) * 4
        ttft = profile.base_ttft_ms + lane * profile.queue_penalty_ms + (index % 3) * 3
        generation_ms = output_tokens / profile.output_tokens_per_second * 1000
        e2e = ttft + generation_ms
        failed = profile.failure_every > 0 and (index + 1) % profile.failure_every == 0
        lane_available[lane] = started + e2e / 1000
        records.append(
            RequestRecord(
                request_id=f"sim-{index + 1:04d}",
                case_id=case.case_id,
                provider="transparent-simulator",
                model=model,
                evidence_type="simulated",
                status="error" if failed else "ok",
                started_offset_ms=started * 1000,
                e2e_ms=e2e,
                ttft_ms=None if failed else ttft,
                input_tokens=48 + index % 7,
                output_tokens=None if failed else output_tokens,
                token_count_source="simulated",
                output_text="" if failed else f"Simulated response for {case.case_id}",
                error="simulated_capacity_error" if failed else None,
                metadata={"lane": lane},
            )
        )
    elapsed = max(lane_available)
    return {
        "schema_version": 1,
        "evidence_type": "simulated",
        "provider": "transparent-simulator",
        "model": model,
        "profile": asdict(profile),
        "summary": summarize_records(records, elapsed_seconds=elapsed, concurrency=concurrency),
        "records": [record.to_dict() for record in records],
        "limitations": [
            "These values exercise the analysis pipeline; they are not model or hardware measurements.",
            "Use the endpoint or local Reka Edge commands to collect observed evidence.",
        ],
    }


def simulate_concurrency_sweep(
    prompts: Sequence[PromptCase],
    *,
    request_count_per_level: int,
    concurrency_levels: Sequence[int],
    model: str,
    profile: SimulatedProfile | None = None,
) -> dict[str, Any]:
    """Exercise the sweep/reporting contract with visibly synthetic values."""

    levels = validate_concurrency_levels(concurrency_levels)
    if request_count_per_level <= 0:
        raise ValueError("request_count_per_level must be positive")
    runs = [
        simulate_benchmark(
            prompts,
            request_count=request_count_per_level,
            concurrency=level,
            model=model,
            profile=profile,
        )
        for level in levels
    ]
    return {
        "schema_version": 1,
        "evidence_type": "simulated",
        "provider": "transparent-simulator",
        "model": model,
        "concurrency_levels": levels,
        "request_count_per_level": request_count_per_level,
        "warmup_requests_per_level": 0,
        "runs": runs,
        "limitations": [
            "This sweep proves the comparison pipeline, not endpoint scaling behavior.",
            "Use the sweep command against an authorized endpoint for observed evidence.",
        ],
    }


def _estimated_token_count(text: str) -> int:
    """Conservative visible fallback when a provider omits token usage."""

    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def _content_for_prompt(case: PromptCase) -> str | list[dict[str, Any]]:
    if case.modality == "text":
        return case.prompt
    media_type = f"{case.modality}_url"
    return [
        {"type": media_type, media_type: {"url": case.media_url}},
        {"type": "text", "text": case.prompt},
    ]


@dataclass(frozen=True)
class EndpointConfig:
    """One OpenAI-compatible endpoint under evaluation."""

    base_url: str
    model: str
    api_key: str
    provider_name: str = "openai-compatible"
    auth_header: str = "authorization"
    timeout_seconds: float = 120.0


class OpenAICompatibleBenchmark:
    """Small streaming benchmark client designed for inspectable POC evidence."""

    def __init__(
        self,
        config: EndpointConfig,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.config = config
        self.transport = transport

    def _headers(self) -> dict[str, str]:
        if self.config.auth_header.lower() == "x-api-key":
            return {"X-Api-Key": self.config.api_key}
        return {"Authorization": f"Bearer {self.config.api_key}"}

    def _api_base_url(self) -> str:
        root = self.config.base_url.rstrip("/")
        return root if root.endswith("/v1") else root + "/v1"

    async def _one_request(
        self,
        client: httpx.AsyncClient,
        case: PromptCase,
        request_id: str,
        benchmark_started: float,
        semaphore: asyncio.Semaphore,
    ) -> RequestRecord:
        async with semaphore:
            started = time.perf_counter()
            first_token_at: float | None = None
            output_parts: list[str] = []
            input_tokens: int | None = None
            output_tokens: int | None = None
            token_source = "estimated"
            try:
                payload = {
                    "model": self.config.model,
                    "messages": [{"role": "user", "content": _content_for_prompt(case)}],
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    "temperature": 0,
                }
                async with client.stream(
                    "POST",
                    "/chat/completions",
                    headers=self._headers(),
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        raw = line[5:].strip()
                        if not raw or raw == "[DONE]":
                            continue
                        event = json.loads(raw)
                        usage = event.get("usage") or {}
                        if usage:
                            input_tokens = usage.get("prompt_tokens")
                            output_tokens = usage.get("completion_tokens")
                            token_source = "provider"
                        choices = event.get("choices") or []
                        if not choices:
                            continue
                        delta = choices[0].get("delta") or {}
                        content = delta.get("content")
                        if content:
                            if first_token_at is None:
                                first_token_at = time.perf_counter()
                            output_parts.append(str(content))
                finished = time.perf_counter()
                output_text = "".join(output_parts)
                if output_tokens is None:
                    output_tokens = _estimated_token_count(output_text)
                return RequestRecord(
                    request_id=request_id,
                    case_id=case.case_id,
                    provider=self.config.provider_name,
                    model=self.config.model,
                    evidence_type="observed-endpoint",
                    status="ok",
                    started_offset_ms=(started - benchmark_started) * 1000,
                    e2e_ms=(finished - started) * 1000,
                    ttft_ms=(first_token_at - started) * 1000 if first_token_at else None,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    token_count_source=token_source,
                    output_text=output_text,
                    metadata={"base_url": self.config.base_url},
                )
            except Exception as exc:  # benchmark evidence must preserve provider failures
                finished = time.perf_counter()
                return RequestRecord(
                    request_id=request_id,
                    case_id=case.case_id,
                    provider=self.config.provider_name,
                    model=self.config.model,
                    evidence_type="observed-endpoint",
                    status="error",
                    started_offset_ms=(started - benchmark_started) * 1000,
                    e2e_ms=(finished - started) * 1000,
                    ttft_ms=None,
                    input_tokens=None,
                    output_tokens=None,
                    token_count_source="unavailable",
                    error=f"{type(exc).__name__}: {exc}",
                    metadata={"base_url": self.config.base_url},
                )

    async def run(
        self, prompts: Sequence[PromptCase], *, request_count: int, concurrency: int
    ) -> dict[str, Any]:
        if not prompts:
            raise ValueError("At least one prompt is required")
        if request_count <= 0 or concurrency <= 0:
            raise ValueError("request_count and concurrency must be positive")
        timeout = httpx.Timeout(self.config.timeout_seconds)
        limits = httpx.Limits(max_connections=max(concurrency, 1))
        benchmark_started = time.perf_counter()
        semaphore = asyncio.Semaphore(concurrency)
        async with httpx.AsyncClient(
            base_url=self._api_base_url(),
            timeout=timeout,
            limits=limits,
            transport=self.transport,
        ) as client:
            tasks = [
                self._one_request(
                    client,
                    prompts[index % len(prompts)],
                    f"req-{index + 1:04d}",
                    benchmark_started,
                    semaphore,
                )
                for index in range(request_count)
            ]
            records = await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - benchmark_started
        return {
            "schema_version": 1,
            "recorded_at_utc": datetime.now(UTC).isoformat(),
            "evidence_type": "observed-endpoint",
            "provider": self.config.provider_name,
            "model": self.config.model,
            "summary": summarize_records(records, elapsed_seconds=elapsed, concurrency=concurrency),
            "records": [record.to_dict() for record in records],
            "limitations": [
                "Client-side timings include network and endpoint behavior visible to this machine.",
                "Capacity conclusions require repeated warm and cold runs at the buyer's traffic shape.",
            ],
        }

    async def run_sweep(
        self,
        prompts: Sequence[PromptCase],
        *,
        request_count_per_level: int,
        concurrency_levels: Sequence[int],
        warmup_requests_per_level: int = 1,
    ) -> dict[str, Any]:
        """Measure several concurrency levels while retaining excluded warm-up evidence."""

        levels = validate_concurrency_levels(concurrency_levels)
        if request_count_per_level <= 0:
            raise ValueError("request_count_per_level must be positive")
        if warmup_requests_per_level < 0:
            raise ValueError("warmup_requests_per_level cannot be negative")
        runs: list[dict[str, Any]] = []
        for level in levels:
            warmup = None
            if warmup_requests_per_level:
                warmup = await self.run(
                    prompts,
                    request_count=warmup_requests_per_level,
                    concurrency=min(level, warmup_requests_per_level),
                )
            measured = await self.run(
                prompts,
                request_count=request_count_per_level,
                concurrency=level,
            )
            measured["warmup"] = {
                "excluded_from_measurement": True,
                "benchmark": warmup,
            }
            runs.append(measured)
        return {
            "schema_version": 1,
            "recorded_at_utc": datetime.now(UTC).isoformat(),
            "evidence_type": "observed-endpoint",
            "provider": self.config.provider_name,
            "model": self.config.model,
            "concurrency_levels": levels,
            "request_count_per_level": request_count_per_level,
            "warmup_requests_per_level": warmup_requests_per_level,
            "runs": runs,
            "limitations": [
                "Warm-up requests are preserved but excluded from each measured summary.",
                "A client-side sweep characterizes only the recorded endpoint, workload, and network path.",
                "Repeat runs are required before making a capacity or service-level commitment.",
            ],
        }

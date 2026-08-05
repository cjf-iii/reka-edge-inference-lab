"""Bounded capacity and public-price estimates from benchmark evidence."""

from __future__ import annotations

import math
from typing import Any

from .models import Pricing, Workload


def _ceil_units(demand: float, observed_capacity: float, target_utilization: float) -> int:
    if observed_capacity <= 0:
        raise ValueError("Observed capacity must be positive")
    return max(1, math.ceil(demand / (observed_capacity * target_utilization)))


def build_capacity_plan(
    benchmark: dict[str, Any], workload: Workload, pricing: Pricing
) -> dict[str, Any]:
    """Translate observed or simulated benchmark evidence into an explicit estimate."""

    summary = benchmark.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("Benchmark summary is missing")
    e2e = summary.get("e2e_ms")
    if not isinstance(e2e, dict):
        raise ValueError("Benchmark contains no successful end-to-end latency evidence")
    observed_request_rps = float(summary.get("request_throughput_per_second", 0))
    observed_output_tps = float(summary.get("output_tokens_per_second", 0))
    if observed_request_rps <= 0 or observed_output_tps <= 0:
        raise ValueError("Benchmark throughput must be positive")

    peak_rps = workload.steady_requests_per_second * workload.peak_multiplier
    peak_output_tps = peak_rps * workload.average_output_tokens
    request_units = _ceil_units(peak_rps, observed_request_rps, workload.target_utilization)
    token_units = _ceil_units(peak_output_tps, observed_output_tps, workload.target_utilization)
    capacity_units = max(request_units, token_units)
    expected_in_flight_at_p95 = peak_rps * (float(e2e["p95"]) / 1000)

    input_cost = (
        workload.monthly_requests
        * workload.average_input_tokens
        / 1_000_000
        * pricing.input_per_million_tokens
    )
    output_cost = (
        workload.monthly_requests
        * workload.average_output_tokens
        / 1_000_000
        * pricing.output_per_million_tokens
    )
    video_index_cost = workload.video_minutes_indexed_per_month * pricing.video_index_per_minute
    video_search_cost = workload.video_searches_per_month * pricing.video_search_each
    monthly_public_unit_cost = input_cost + output_cost + video_index_cost + video_search_cost

    observed_p95 = float(e2e["p95"])
    recommendations: list[str] = []
    if observed_p95 > workload.target_p95_ms:
        recommendations.append(
            "The tested p95 exceeds the target; reduce work per request, change the serving tier, "
            "or evaluate additional capacity before a production commitment."
        )
    else:
        recommendations.append(
            "The tested p95 is inside the target for this run; repeat across warm, cold, burst, and "
            "failure conditions before treating it as an SLO."
        )
    if workload.privacy_requirement.lower() not in {"standard", "public-api"}:
        recommendations.append(
            "Validate retention, regional processing, private endpoint, and customer-managed deployment "
            "requirements with security stakeholders."
        )
    if capacity_units > 1 or workload.peak_multiplier >= 3:
        recommendations.append(
            "The peak profile warrants a reserved-capacity or dedicated-endpoint conversation; public "
            "unit pricing alone does not establish availability."
        )
    if workload.deployment_preference.lower() in {"local", "on-device", "on-premises"}:
        recommendations.append(
            "Run the same golden set on local Reka Edge and a hosted endpoint; compare quality, latency, "
            "operations, and license constraints rather than assuming local is automatically cheaper."
        )

    return {
        "schema_version": 1,
        "evidence_type": benchmark.get("evidence_type", "unknown"),
        "benchmark_provider": benchmark.get("provider", "unknown"),
        "benchmark_model": benchmark.get("model", "unknown"),
        "workload": workload.to_dict(),
        "pricing": pricing.to_dict(),
        "demand": {
            "steady_requests_per_second": workload.steady_requests_per_second,
            "peak_requests_per_second": peak_rps,
            "peak_output_tokens_per_second": peak_output_tps,
            "expected_in_flight_at_observed_p95": expected_in_flight_at_p95,
        },
        "observed_capacity_unit": {
            "request_throughput_per_second": observed_request_rps,
            "output_tokens_per_second": observed_output_tps,
            "p95_e2e_ms": observed_p95,
            "benchmark_concurrency": summary.get("concurrency"),
        },
        "estimate": {
            "units_by_request_throughput": request_units,
            "units_by_output_token_throughput": token_units,
            "estimated_capacity_units": capacity_units,
            "target_utilization": workload.target_utilization,
            "target_p95_ms": workload.target_p95_ms,
            "observed_p95_meets_target": observed_p95 <= workload.target_p95_ms,
        },
        "monthly_public_unit_cost": {
            "input_tokens": input_cost,
            "output_tokens": output_cost,
            "video_indexing": video_index_cost,
            "video_search": video_search_cost,
            "total": monthly_public_unit_cost,
            "currency": "USD",
            "excludes": [
                "retries and failed work",
                "storage not represented in the public price file",
                "private endpoints or dedicated capacity",
                "engineering and operations",
                "taxes and negotiated discounts",
            ],
        },
        "discovery_questions": [
            "What request and media mix must the evaluation reproduce?",
            "Which quality failures are disqualifying on the buyer's golden set?",
            "What are the p95 and p99 targets at steady load and peak burst?",
            "Where may prompts, media, embeddings, logs, and outputs be stored or processed?",
            "Which work can run asynchronously, and which interactions require streaming?",
            "What utilization and commitment risk is acceptable for reserved or dedicated capacity?",
            "Who signs off on quality, security, integration, and the commercial case?",
        ],
        "recommendations": recommendations,
        "limitations": [
            "A benchmarked endpoint is treated as one capacity unit; it is not assumed to equal one GPU.",
            "The estimate extrapolates from the tested prompt mix and concurrency and is not an SLA.",
            "Public prices are dated inputs and must be rechecked before a buyer-facing proposal.",
            "Private capacity and negotiated enterprise terms require a vendor quote.",
        ],
    }

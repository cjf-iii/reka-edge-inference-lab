"""Command-line interface for discovery, benchmarking, planning, and verification."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from .artifacts import verify_bundle
from .benchmark import (
    EndpointConfig,
    OpenAICompatibleBenchmark,
    SimulatedProfile,
    simulate_benchmark,
    simulate_concurrency_sweep,
    validate_concurrency_levels,
)
from .capacity import build_capacity_plan
from .local_edge import LocalEdgeSession, system_probe
from .media import create_demo_image
from .models import Pricing, PromptCase, Workload, read_json, write_json
from .report import write_report_bundle


def _load_prompts(path: str | Path) -> list[PromptCase]:
    data = read_json(path)
    rows = data.get("prompts")
    if not isinstance(rows, list):
        raise ValueError("Prompt file must contain a prompts list")
    return [PromptCase.from_mapping(row) for row in rows]


def _load_inputs(
    prompts_path: str | Path, workload_path: str | Path, pricing_path: str | Path
) -> tuple[list[PromptCase], Workload, Pricing]:
    return (
        _load_prompts(prompts_path),
        Workload.from_mapping(read_json(workload_path)),
        Pricing.from_mapping(read_json(pricing_path)),
    )


def command_demo(args: argparse.Namespace) -> int:
    prompts, workload, pricing = _load_inputs(args.prompts, args.workload, args.pricing)
    output = Path(args.output)
    create_demo_image(output / "owned_multimodal_card.png")
    benchmark = simulate_benchmark(
        prompts,
        request_count=args.requests,
        concurrency=args.concurrency,
        model="reka-edge-evaluation-shape",
        profile=SimulatedProfile(failure_every=args.failure_every),
    )
    plan = build_capacity_plan(benchmark, workload, pricing)
    paths = write_report_bundle(output, benchmark, plan)
    sweep = simulate_concurrency_sweep(
        prompts,
        request_count_per_level=args.requests,
        concurrency_levels=(1, 2, 4, 8),
        model="reka-edge-evaluation-shape",
        profile=SimulatedProfile(failure_every=args.failure_every),
    )
    sweep_path = output / "concurrency_sweep.json"
    write_json(sweep_path, sweep)
    paths["concurrency_sweep"] = str(sweep_path)
    result = verify_bundle(output)
    print(json.dumps({"artifacts": paths, "verification": result}, indent=2))
    return 0


def command_endpoint(args: argparse.Namespace) -> int:
    prompts, workload, pricing = _load_inputs(args.prompts, args.workload, args.pricing)
    api_key = os.environ.get(args.api_key_env)
    if not api_key and args.base_url.startswith(("http://127.0.0.1", "http://localhost")):
        api_key = "unused-local-key"
    if not api_key:
        raise ValueError(f"{args.api_key_env} is not set")
    client = OpenAICompatibleBenchmark(
        EndpointConfig(
            base_url=args.base_url,
            model=args.model,
            api_key=api_key,
            provider_name=args.provider_name,
            auth_header=args.auth_header,
            timeout_seconds=args.timeout,
        )
    )
    benchmark = asyncio.run(
        client.run(prompts, request_count=args.requests, concurrency=args.concurrency)
    )
    plan = build_capacity_plan(benchmark, workload, pricing)
    paths = write_report_bundle(args.output, benchmark, plan)
    result = verify_bundle(args.output)
    print(json.dumps({"artifacts": paths, "verification": result}, indent=2))
    return 0


def command_sweep(args: argparse.Namespace) -> int:
    prompts = _load_prompts(args.prompts)
    api_key = os.environ.get(args.api_key_env)
    if not api_key and args.base_url.startswith(("http://127.0.0.1", "http://localhost")):
        api_key = "unused-local-key"
    if not api_key:
        raise ValueError(f"{args.api_key_env} is not set")
    client = OpenAICompatibleBenchmark(
        EndpointConfig(
            base_url=args.base_url,
            model=args.model,
            api_key=api_key,
            provider_name=args.provider_name,
            auth_header=args.auth_header,
            timeout_seconds=args.timeout,
        )
    )
    sweep = asyncio.run(
        client.run_sweep(
            prompts,
            request_count_per_level=args.requests_per_level,
            concurrency_levels=args.levels,
            warmup_requests_per_level=args.warmup_requests,
        )
    )
    write_json(args.output, sweep)
    comparison = [
        {
            "concurrency": run["summary"]["concurrency"],
            "success_rate": run["summary"]["success_rate"],
            "request_throughput_per_second": run["summary"]["request_throughput_per_second"],
            "p95_e2e_ms": (run["summary"].get("e2e_ms") or {}).get("p95"),
            "p95_ttft_ms": (run["summary"].get("ttft_ms") or {}).get("p95"),
        }
        for run in sweep["runs"]
    ]
    print(json.dumps({"output": args.output, "comparison": comparison}, indent=2))
    return 0


def command_probe(_args: argparse.Namespace) -> int:
    print(json.dumps(system_probe(), indent=2))
    return 0


def command_make_media(args: argparse.Namespace) -> int:
    created = create_demo_image(args.output)
    print(created)
    return 0


def command_local(args: argparse.Namespace) -> int:
    if not args.accept_model_license:
        raise ValueError(
            "Review the Reka Edge model license, then pass --accept-model-license to run the pinned model."
        )
    image_path = Path(args.image)
    if not image_path.exists() and args.create_demo_image:
        create_demo_image(image_path)
    session = LocalEdgeSession()
    benchmark = session.run(
        prompt=args.prompt,
        image_path=image_path if image_path.exists() else None,
        max_new_tokens=args.max_new_tokens,
    )
    write_json(args.output, benchmark)
    print(json.dumps({"output": args.output, "summary": benchmark["summary"]}, indent=2))
    return 0


def command_verify(args: argparse.Namespace) -> int:
    print(json.dumps(verify_bundle(args.directory), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reka-infer",
        description="Reproducible multimodal inference evaluation for buyer discovery.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="Generate a deterministic offline report bundle")
    demo.add_argument("--prompts", default="examples/prompts.json")
    demo.add_argument("--workload", default="examples/workload.json")
    demo.add_argument("--pricing", default="examples/reka_public_pricing_2026-08-04.json")
    demo.add_argument("--output", default="artifacts/demo")
    demo.add_argument("--requests", type=int, default=24)
    demo.add_argument("--concurrency", type=int, default=4)
    demo.add_argument("--failure-every", type=int, default=0)
    demo.set_defaults(func=command_demo)

    endpoint = subparsers.add_parser(
        "endpoint", help="Benchmark an OpenAI-compatible streaming endpoint"
    )
    endpoint.add_argument("--base-url", default="https://api.reka.ai")
    endpoint.add_argument("--model", default="reka-edge")
    endpoint.add_argument("--provider-name", default="reka-api")
    endpoint.add_argument("--api-key-env", default="REKA_API_KEY")
    endpoint.add_argument(
        "--auth-header", choices=("authorization", "x-api-key"), default="authorization"
    )
    endpoint.add_argument("--prompts", default="examples/prompts.json")
    endpoint.add_argument("--workload", default="examples/workload.json")
    endpoint.add_argument("--pricing", default="examples/reka_public_pricing_2026-08-04.json")
    endpoint.add_argument("--output", default="artifacts/endpoint")
    endpoint.add_argument("--requests", type=int, default=12)
    endpoint.add_argument("--concurrency", type=int, default=2)
    endpoint.add_argument("--timeout", type=float, default=120)
    endpoint.set_defaults(func=command_endpoint)

    sweep = subparsers.add_parser(
        "sweep", help="Measure an endpoint across several concurrency levels"
    )
    sweep.add_argument("--base-url", default="https://api.reka.ai")
    sweep.add_argument("--model", default="reka-edge")
    sweep.add_argument("--provider-name", default="reka-api")
    sweep.add_argument("--api-key-env", default="REKA_API_KEY")
    sweep.add_argument(
        "--auth-header", choices=("authorization", "x-api-key"), default="authorization"
    )
    sweep.add_argument("--prompts", default="examples/prompts.json")
    sweep.add_argument("--output", default="artifacts/concurrency_sweep.json")
    sweep.add_argument("--requests-per-level", type=int, default=12)
    sweep.add_argument(
        "--levels",
        type=lambda value: validate_concurrency_levels(
            [int(part.strip()) for part in value.split(",") if part.strip()]
        ),
        default=[1, 2, 4],
        help="Comma-separated concurrency levels, for example 1,2,4",
    )
    sweep.add_argument("--warmup-requests", type=int, default=1)
    sweep.add_argument("--timeout", type=float, default=120)
    sweep.set_defaults(func=command_sweep)

    probe = subparsers.add_parser("probe", help="Check the documented local hardware boundary")
    probe.set_defaults(func=command_probe)

    make_media = subparsers.add_parser("make-media", help="Create the owned evaluation image")
    make_media.add_argument("--output", default="artifacts/owned_multimodal_card.png")
    make_media.set_defaults(func=command_make_media)

    local = subparsers.add_parser(
        "local", help="Run the pinned Reka Edge model through Transformers and Apple MPS"
    )
    local.add_argument("--image", default="artifacts/owned_multimodal_card.png")
    local.add_argument("--create-demo-image", action="store_true")
    local.add_argument(
        "--prompt",
        default=(
            "List the three colored shapes from left to right and copy the request ID exactly. "
            "Return one compact sentence."
        ),
    )
    local.add_argument("--max-new-tokens", type=int, default=96)
    local.add_argument("--output", default="artifacts/local_reka_edge.json")
    local.add_argument("--accept-model-license", action="store_true")
    local.set_defaults(func=command_local)

    verify = subparsers.add_parser("verify", help="Fail closed on an incomplete report bundle")
    verify.add_argument("directory", nargs="?", default="artifacts/demo")
    verify.set_defaults(func=command_verify)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

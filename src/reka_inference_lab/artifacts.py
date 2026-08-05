"""Fail-closed verification for generated public evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import read_json


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_bundle(directory: str | Path) -> dict[str, Any]:
    root = Path(directory)
    required = {
        "benchmark": root / "benchmark.json",
        "capacity_plan": root / "capacity_plan.json",
        "markdown_report": root / "decision_report.md",
        "html_report": root / "decision_report.html",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    _require(not missing, f"Missing artifacts: {', '.join(missing)}")
    benchmark = read_json(required["benchmark"])
    plan = read_json(required["capacity_plan"])
    summary = benchmark.get("summary")
    _require(isinstance(summary, dict), "Benchmark summary is missing")
    _require(summary.get("request_count", 0) > 0, "No benchmark request evidence")
    _require(0 <= summary.get("success_rate", -1) <= 1, "Invalid success rate")
    _require(
        summary.get("success_count", 0) + summary.get("failure_count", 0)
        == summary.get("request_count"),
        "Request totals do not reconcile",
    )
    records = benchmark.get("records")
    _require(isinstance(records, list), "Benchmark records are missing")
    _require(len(records) == summary["request_count"], "Raw record count does not match summary")
    _require(
        plan.get("evidence_type") == benchmark.get("evidence_type"),
        "Plan evidence type does not match benchmark",
    )
    estimate = plan.get("estimate")
    _require(isinstance(estimate, dict), "Capacity estimate is missing")
    _require(estimate.get("estimated_capacity_units", 0) >= 1, "Invalid capacity estimate")
    pricing = plan.get("pricing")
    _require(isinstance(pricing, dict), "Pricing source is missing")
    _require(bool(pricing.get("checked_at")), "Pricing checked_at is missing")
    _require(str(pricing.get("source_url", "")).startswith("https://"), "Invalid pricing URL")
    markdown = required["markdown_report"].read_text(encoding="utf-8")
    rendered_html = required["html_report"].read_text(encoding="utf-8")
    _require(
        f"Evidence type: `{benchmark['evidence_type']}`" in markdown,
        "Report evidence label missing",
    )
    _require("quote required" in rendered_html, "Enterprise quote boundary missing")
    _require(pricing["source_url"] in markdown, "Pricing source link missing from Markdown")
    sweep_path = root / "concurrency_sweep.json"
    if sweep_path.is_file():
        sweep = read_json(sweep_path)
        _require(
            sweep.get("evidence_type") == benchmark.get("evidence_type"),
            "Sweep evidence type does not match benchmark",
        )
        runs = sweep.get("runs")
        _require(isinstance(runs, list) and len(runs) >= 2, "Sweep requires at least two runs")
        levels = sweep.get("concurrency_levels")
        _require(
            isinstance(levels, list) and len(levels) == len(runs),
            "Sweep levels do not match runs",
        )
        for level, run in zip(levels, runs, strict=True):
            run_summary = run.get("summary")
            _require(isinstance(run_summary, dict), "Sweep run summary is missing")
            _require(run_summary.get("concurrency") == level, "Sweep concurrency mismatch")
            run_records = run.get("records")
            _require(
                isinstance(run_records, list)
                and len(run_records) == run_summary.get("request_count"),
                "Sweep raw record count does not match summary",
            )
    return {
        "ok": True,
        "evidence_type": benchmark["evidence_type"],
        "request_count": summary["request_count"],
        "success_count": summary["success_count"],
        "estimated_capacity_units": estimate["estimated_capacity_units"],
        "pricing_checked_at": pricing["checked_at"],
        "concurrency_levels": sweep.get("concurrency_levels") if sweep_path.is_file() else None,
    }

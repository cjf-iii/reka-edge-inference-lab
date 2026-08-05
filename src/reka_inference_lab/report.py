"""Generate developer, ML-lead, and executive views from the same evidence."""

from __future__ import annotations

import html
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .models import write_json


def _money(value: float) -> str:
    return f"${value:,.2f}"


def _metric(distribution: dict[str, Any] | None, key: str) -> str:
    if not distribution:
        return "n/a"
    return f"{float(distribution[key]):,.1f} ms"


def markdown_report(benchmark: dict[str, Any], plan: dict[str, Any]) -> str:
    summary = benchmark["summary"]
    estimate = plan["estimate"]
    cost = plan["monthly_public_unit_cost"]
    observed = plan["observed_capacity_unit"]
    evidence = str(benchmark.get("evidence_type", "unknown"))
    warning = (
        "This is synthetic pipeline evidence, not measured inference."
        if evidence == "simulated"
        else "These timings were observed from this client and tested endpoint/configuration."
    )
    questions = "\n".join(
        f"{index}. {question}" for index, question in enumerate(plan["discovery_questions"], 1)
    )
    recommendations = "\n".join(f"- {item}" for item in plan["recommendations"])
    limitations = "\n".join(f"- {item}" for item in plan["limitations"])
    return f"""# Inference Evaluation Decision Report

*Generated: {datetime.now(UTC).isoformat()} | Evidence type: `{evidence}`*

> {warning}

## Workload

| Field | Value |
|---|---:|
| Scenario | {plan["workload"]["name"]} |
| Modality | {plan["workload"]["modality"]} |
| Steady requests/s | {plan["demand"]["steady_requests_per_second"]:.3f} |
| Peak requests/s | {plan["demand"]["peak_requests_per_second"]:.3f} |
| Target p95 | {plan["workload"]["target_p95_ms"]:,.0f} ms |
| Target utilization | {plan["workload"]["target_utilization"]:.0%} |

## Benchmark evidence

| Metric | Result |
|---|---:|
| Success rate | {summary["success_rate"]:.1%} |
| Concurrency | {summary["concurrency"]} |
| Request throughput | {summary["request_throughput_per_second"]:.3f}/s |
| Output throughput | {summary["output_tokens_per_second"]:.2f} tokens/s |
| p50 end-to-end | {_metric(summary["e2e_ms"], "p50")} |
| p95 end-to-end | {_metric(summary["e2e_ms"], "p95")} |
| p99 end-to-end | {_metric(summary["e2e_ms"], "p99")} |
| p50 TTFT | {_metric(summary["ttft_ms"], "p50")} |
| p95 TTFT | {_metric(summary["ttft_ms"], "p95")} |

## Capacity estimate

| Field | Estimate |
|---|---:|
| Benchmark capacity units | {estimate["estimated_capacity_units"]} |
| Units from request demand | {estimate["units_by_request_throughput"]} |
| Units from token demand | {estimate["units_by_output_token_throughput"]} |
| Expected in-flight at observed p95 | {plan["demand"]["expected_in_flight_at_observed_p95"]:.2f} |
| Observed p95 meets target | {"yes" if estimate["observed_p95_meets_target"] else "no"} |

One capacity unit means one endpoint/configuration matching the benchmark. It is **not** assumed to equal one GPU. The observed capacity unit delivered {observed["request_throughput_per_second"]:.3f} successful requests/s and {observed["output_tokens_per_second"]:.2f} output tokens/s in this run.

## Public-unit monthly cost estimate

| Component | Estimate |
|---|---:|
| Input tokens | {_money(cost["input_tokens"])} |
| Output tokens | {_money(cost["output_tokens"])} |
| Video indexing | {_money(cost["video_indexing"])} |
| Video search | {_money(cost["video_search"])} |
| **Total represented public units** | **{_money(cost["total"])}** |

Pricing source: [{plan["pricing"]["name"]}]({plan["pricing"]["source_url"]}) checked {plan["pricing"]["checked_at"]}. This excludes negotiated enterprise capacity and the other exclusions preserved in `capacity_plan.json`.

## Discovery questions

{questions}

## Recommendations

{recommendations}

## Limitations

{limitations}
"""


def html_report(benchmark: dict[str, Any], plan: dict[str, Any]) -> str:
    summary = benchmark["summary"]
    estimate = plan["estimate"]
    cost = plan["monthly_public_unit_cost"]
    evidence = html.escape(str(benchmark.get("evidence_type", "unknown")))
    warning = (
        "Synthetic pipeline evidence — not a model or hardware measurement."
        if evidence == "simulated"
        else "Observed client-side evidence for the tested endpoint and configuration."
    )
    question_items = "".join(
        f"<li>{html.escape(question)}</li>" for question in plan["discovery_questions"]
    )
    recommendation_items = "".join(
        f"<li>{html.escape(item)}</li>" for item in plan["recommendations"]
    )
    source_url = html.escape(plan["pricing"]["source_url"], quote=True)
    workload_name = html.escape(plan["workload"]["name"])
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Inference Evaluation — {workload_name}</title>
  <style>
    :root {{ color-scheme: dark; --bg:#0f172a; --panel:#172033; --line:#334155; --text:#e2e8f0; --muted:#94a3b8; --blue:#60a5fa; --green:#34d399; --amber:#fbbf24; }}
    * {{ box-sizing:border-box; }} body {{ margin:0; background:linear-gradient(145deg,#0b1120,#111827); color:var(--text); font:16px/1.55 ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
    main {{ max-width:1120px; margin:auto; padding:48px 24px 72px; }} h1 {{ font-size:clamp(2rem,5vw,4rem); line-height:1; margin:.2em 0; letter-spacing:-.04em; }} h2 {{ margin-top:2.2rem; }} .eyebrow {{ color:var(--blue); font-weight:700; text-transform:uppercase; letter-spacing:.12em; }}
    .notice {{ border:1px solid var(--amber); background:#2a2417; padding:14px 18px; border-radius:12px; }} .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:14px; margin:22px 0; }}
    .card {{ background:rgba(23,32,51,.92); border:1px solid var(--line); border-radius:16px; padding:20px; }} .value {{ display:block; color:var(--green); font-size:1.8rem; font-weight:750; letter-spacing:-.03em; }} .label {{ color:var(--muted); font-size:.9rem; }}
    table {{ width:100%; border-collapse:collapse; background:rgba(23,32,51,.75); }} th,td {{ border-bottom:1px solid var(--line); padding:12px; text-align:left; }} td:last-child,th:last-child {{ text-align:right; }} a {{ color:var(--blue); }} li {{ margin:.45rem 0; }} footer {{ color:var(--muted); margin-top:3rem; }}
  </style>
</head>
<body><main>
  <div class="eyebrow">Buyer evaluation evidence · {evidence}</div>
  <h1>{workload_name}</h1>
  <p class="notice">{html.escape(warning)}</p>
  <section class="grid">
    <div class="card"><span class="label">Success rate</span><span class="value">{summary["success_rate"]:.1%}</span></div>
    <div class="card"><span class="label">p95 end-to-end</span><span class="value">{_metric(summary["e2e_ms"], "p95")}</span></div>
    <div class="card"><span class="label">Output throughput</span><span class="value">{summary["output_tokens_per_second"]:.1f} tok/s</span></div>
    <div class="card"><span class="label">Capacity units</span><span class="value">{estimate["estimated_capacity_units"]}</span></div>
  </section>
  <h2>Decision frame</h2>
  <table><tr><th>Signal</th><th>Result</th></tr>
    <tr><td>Peak request demand</td><td>{plan["demand"]["peak_requests_per_second"]:.3f}/s</td></tr>
    <tr><td>Target / observed p95</td><td>{plan["workload"]["target_p95_ms"]:,.0f} / {plan["observed_capacity_unit"]["p95_e2e_ms"]:,.1f} ms</td></tr>
    <tr><td>Public-unit monthly estimate</td><td>{_money(cost["total"])}</td></tr>
    <tr><td>Private capacity price</td><td>quote required</td></tr>
  </table>
  <h2>Questions that decide the deal</h2><ol>{question_items}</ol>
  <h2>Recommendations</h2><ul>{recommendation_items}</ul>
  <footer>Pricing checked {html.escape(plan["pricing"]["checked_at"])}: <a href="{source_url}">{html.escape(plan["pricing"]["name"])}</a>. Capacity units are benchmark configurations, not assumed GPUs. See the JSON evidence for exclusions and limitations.</footer>
</main></body></html>
"""


def write_report_bundle(
    output_directory: str | Path,
    benchmark: dict[str, Any],
    plan: dict[str, Any],
) -> dict[str, str]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    benchmark_path = output / "benchmark.json"
    plan_path = output / "capacity_plan.json"
    markdown_path = output / "decision_report.md"
    html_path = output / "decision_report.html"
    write_json(benchmark_path, benchmark)
    write_json(plan_path, plan)
    markdown_path.write_text(markdown_report(benchmark, plan), encoding="utf-8")
    html_path.write_text(html_report(benchmark, plan), encoding="utf-8")
    return {
        "benchmark": str(benchmark_path),
        "capacity_plan": str(plan_path),
        "markdown_report": str(markdown_path),
        "html_report": str(html_path),
    }

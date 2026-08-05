# Verification record

Status: local Reka Edge execution and clean-archive release gate verified.

## Observed Reka Edge smoke test

Recorded August 4, 2026 PDT (`2026-08-05T05:11:15Z`) with:

```bash
uv sync --locked --group dev --group notebook --group local
uv run reka-infer local \
  --create-demo-image \
  --accept-model-license \
  --output evidence/observed_local_smoke_2026-08-04.json
```

| Field | Observed value |
|---|---:|
| Model | `RekaAI/reka-edge-2603` |
| Revision | `492c81c225fbf5f3263a8245b00827721b119a13` |
| Runtime | Transformers 4.57.3, float16, Apple MPS |
| Mac | arm64, macOS 26.6, 128 GiB unified memory |
| Requests | 1 successful, 0 failed |
| End-to-end | 2,147.5 ms |
| TTFT | 1,158.9 ms |
| Output rate | 9.78 tokens/s |
| Model load | 9.27 s after weights were cached |

The model described the owned test image as: “Blue square, Red circle, Yellow triangle. Request ID: REKA-EDGE-LAB-001.” The record stores a relative media path and SHA-256 rather than an absolute user path.

Raw evidence: [`evidence/observed_local_smoke_2026-08-04.json`](evidence/observed_local_smoke_2026-08-04.json), file SHA-256 `ed4f7ead16f9c76e8653686aafe227d65f0227a3a6af0d531c1af9d55d2a4045`.

This is a one-request compatibility smoke test after the model weights were cached. It proves the pinned Transformers/MPS path executed on the reported machine; it is not a production benchmark, capacity estimate, vendor comparison, or SLA.

## Local release gate

The following passed in the working repository:

```bash
uv run ruff check .
uv run ruff format --check .
uv run python -m unittest discover -s tests -v
uv run --group notebook jupyter nbconvert \
  --to notebook --execute notebooks/01_workload_to_decision.ipynb \
  --output /tmp/reka-workload-to-decision.executed.ipynb
uv run reka-infer demo --output artifacts/demo
uv run reka-infer verify artifacts/demo
```

- Ruff: passed.
- Unit and integration tests: 19 passed.
- Notebook: executed without a failed cell.
- Offline demo: 24 simulated report requests plus a 96-record, four-level simulated concurrency sweep reconciled; generated JSON, Markdown, HTML, and owned image artifacts.
- Artifact verifier: passed with `evidence_type=simulated`, concurrency levels 1/2/4/8, and pricing checked August 4, 2026.
- Rendered HTML: visually inspected at 1440 × 1800; the simulation warning, metrics, capacity boundary, discovery questions, and recommendations were legible.

## Clean-archive release gate

Commit `bf637d1ceacc333f420119a32683ce254397503c` was exported with `git archive` into an empty temporary directory on August 4, 2026 PDT. From that archive—not the development working tree—the following passed:

- `uv sync --locked --group dev --group notebook` with CPython 3.12.13;
- Ruff lint and format checks;
- all 19 unit and integration tests;
- headless execution of `notebooks/01_workload_to_decision.ipynb`;
- the 24-request offline report demo and the 96-record concurrency sweep at levels 1/2/4/8;
- the fail-closed artifact verifier and executed-notebook JSON parse.

The committed file modes were inspected: the pre-push hook and artifact-verification script are executable; source, evidence, documentation, notebooks, workflows, and configuration are regular files. The staged-tree scan found no absolute user path, private résumé, model weight, private key, or credential. The README contains the intentional placeholder `REKA_API_KEY="your-key"`.

## Publication boundary

The local repository has no public remote yet. Public-current status requires the reviewed commit to be visible on GitHub and its GitHub Actions gate to pass. A live Reka API sweep also requires the candidate's own authorized API key; it is not part of the credential-free release gate.

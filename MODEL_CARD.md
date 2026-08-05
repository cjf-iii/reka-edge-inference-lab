# Model and evaluation card

## Intended use

This repository demonstrates how to structure a buyer-facing multimodal inference evaluation. Its default release gate uses a transparent simulator so reviewers can inspect the complete benchmark-to-report path without credentials, a large model download, or special hardware.

The optional local path runs the pinned `RekaAI/reka-edge-2603` revision through Transformers and Apple MPS. It is intended for technical learning and reproducible portfolio evidence on eligible Apple Silicon hardware.

## Evidence classes

- `simulated`: deterministic scheduling values that test analysis logic; never model performance.
- `observed-endpoint`: client-side timings and responses from one OpenAI-compatible endpoint/configuration.
- `observed-local`: timings and responses from the pinned local model on the reported Mac/configuration.
- `vendor-reported`: claims linked to a vendor source but not reproduced here.

## Limitations

- One prompt or one machine does not establish production quality, capacity, or an SLA.
- Client-side timing includes network behavior for remote endpoints.
- Estimated token counts are marked when a provider omits usage data.
- The capacity planner extrapolates from the tested request mix and treats one tested endpoint/configuration as one abstract capacity unit—not one GPU.
- Public prices are dated inputs. Enterprise terms, private capacity, storage, support, and negotiated discounts require a current quote.
- The owned demonstration image is intentionally simple and is not a meaningful multimodal benchmark.

## License and supply-chain boundary

The repository code is MIT licensed. Reka Edge uses a separate custom license that reviewers must read before downloading or running the model. The optional command requires an explicit `--accept-model-license` flag. Model weights are never committed.

The model requires `trust_remote_code=True`. Reproducibility is bounded by the pinned model revision `492c81c225fbf5f3263a8245b00827721b119a13`; reviewers should still inspect that revision's Python files before execution.

## Deployment boundary

The macOS path uses Transformers with MPS and float16. It does not claim to run the official `vllm-reka` server, which Reka documents as a Linux/CUDA path for high-throughput serving.

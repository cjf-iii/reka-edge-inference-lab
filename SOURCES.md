# Sources and design lineage

Checked August 4, 2026.

## Reka sources

- [Reka Edge model card](https://huggingface.co/RekaAI/reka-edge-2603) — pinned model, macOS/MPS path, deployment notes, benchmark claims, and custom model license.
- [Reka API quickstart](https://docs.reka.ai/quickstart) — OpenAI-compatible API and macOS versus Linux/CUDA deployment boundary.
- [Reka API pricing](https://docs.reka.ai/pricing) — dated Chat and Research public unit prices.
- [Reka Vision pricing](https://docs.reka.ai/vision/pricing) — dated video indexing, search, Q&A, and clip units.
- [vllm-reka](https://github.com/reka-ai/vllm-reka) — official Linux/CUDA high-throughput server and its OpenAI-compatible interface.
- [Reka Quant](https://github.com/reka-ai/rekaquant) — low-bit quantization and GGUF design context.
- [Vibe-Eval](https://github.com/reka-ai/reka-vibe-eval) — multimodal evaluation methodology.
- [Research-Eval](https://github.com/reka-ai/research-eval) — generate, judge, and report evaluation shape.

## Inference measurement references

- [AIPerf](https://github.com/ai-dynamo/aiperf) and [NVIDIA GenAI-Perf](https://github.com/triton-inference-server/perf_analyzer) informed the metric vocabulary: TTFT, inter-token latency, end-to-end latency, throughput, concurrency, and raw records.
- [PagedAttention](https://arxiv.org/abs/2309.06180) and [SGLang](https://arxiv.org/abs/2312.07104) informed the batching, queueing, and KV-cache explanations.

## Reuse boundary

This repository is an original, compact implementation. It borrows public measurement concepts and interface contracts, not source code. Model weights and remote model code are not redistributed. The code in this repository is MIT licensed; the Reka Edge weights and model files remain governed by Reka's separate license.

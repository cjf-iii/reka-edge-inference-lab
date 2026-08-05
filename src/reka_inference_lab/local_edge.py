"""Pinned, opt-in Reka Edge execution through Transformers and Apple MPS."""

from __future__ import annotations

import hashlib
import importlib.util
import platform
import subprocess
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .benchmark import summarize_records
from .models import RequestRecord

MODEL_ID = "RekaAI/reka-edge-2603"
MODEL_REVISION = "492c81c225fbf5f3263a8245b00827721b119a13"
MODEL_SOURCE = "https://huggingface.co/RekaAI/reka-edge-2603"


def _media_evidence(image_path: str | Path | None) -> dict[str, str | None]:
    """Describe media reproducibly without publishing an absolute user path."""

    if image_path is None:
        return {
            "image_path": None,
            "image_path_scope": None,
            "image_sha256": None,
        }
    resolved = Path(image_path).expanduser().resolve()
    digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
    try:
        display_path = str(resolved.relative_to(Path.cwd().resolve()))
        scope = "working-directory-relative"
    except ValueError:
        display_path = resolved.name
        scope = "basename-only"
    return {
        "image_path": display_path,
        "image_path_scope": scope,
        "image_sha256": digest,
    }


def _sysctl_int(name: str) -> int | None:
    try:
        result = subprocess.run(
            ["/usr/sbin/sysctl", "-n", name],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return int(result.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def system_probe() -> dict[str, Any]:
    """Report only the hardware facts needed to decide whether to attempt the model."""

    memory_bytes = _sysctl_int("hw.memsize")
    memory_gib = memory_bytes / 1024**3 if memory_bytes else None
    architecture = platform.machine()
    operating_system = platform.system()
    macos_version = platform.mac_ver()[0] or None
    hardware_eligible = (
        operating_system == "Darwin"
        and architecture == "arm64"
        and memory_gib is not None
        and memory_gib >= 32
    )
    return {
        "operating_system": operating_system,
        "macos_version": macos_version,
        "architecture": architecture,
        "memory_gib": round(memory_gib, 1) if memory_gib is not None else None,
        "hardware_eligible": hardware_eligible,
        "torch_installed": importlib.util.find_spec("torch") is not None,
        "transformers_installed": importlib.util.find_spec("transformers") is not None,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_source": MODEL_SOURCE,
        "boundary": (
            "This path uses Transformers with Apple MPS. The official vllm-reka server is a "
            "separate Linux/CUDA deployment path."
        ),
    }


class LocalEdgeSession:
    """Load the pinned model once, then execute inspectable local requests."""

    def __init__(self) -> None:
        probe = system_probe()
        if not probe["hardware_eligible"]:
            raise RuntimeError(
                "Local Reka Edge requires Apple Silicon macOS with at least 32 GiB unified memory."
            )
        try:
            import torch
            from transformers import (
                AutoModelForImageTextToText,
                AutoProcessor,
                TextIteratorStreamer,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Local dependencies are missing. Run: uv sync --locked --group local"
            ) from exc
        if not torch.backends.mps.is_available():
            raise RuntimeError("PyTorch MPS is not available in this environment")
        self.torch = torch
        self.TextIteratorStreamer = TextIteratorStreamer
        load_started = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            trust_remote_code=True,
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            trust_remote_code=True,
            dtype=torch.float16,
        ).eval()
        self.model = self.model.to(torch.device("mps"))
        self.load_seconds = time.perf_counter() - load_started

    def _inputs(self, prompt: str, image_path: str | Path | None) -> dict[str, Any]:
        content: list[dict[str, Any]] = []
        if image_path is not None:
            resolved = Path(image_path).expanduser().resolve()
            if not resolved.is_file():
                raise FileNotFoundError(resolved)
            content.append({"type": "image", "image": str(resolved)})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        device = self.torch.device("mps")
        for key, value in inputs.items():
            if isinstance(value, self.torch.Tensor):
                if value.is_floating_point():
                    inputs[key] = value.to(device=device, dtype=self.torch.float16)
                else:
                    inputs[key] = value.to(device=device)
        return inputs

    def run(
        self,
        *,
        prompt: str,
        image_path: str | Path | None,
        max_new_tokens: int,
    ) -> dict[str, Any]:
        if not prompt.strip():
            raise ValueError("prompt cannot be empty")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        inputs = self._inputs(prompt, image_path)
        streamer = self.TextIteratorStreamer(
            self.processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=600,
        )
        sep_token_id = self.processor.tokenizer.convert_tokens_to_ids("<sep>")
        eos_token_ids = [self.processor.tokenizer.eos_token_id]
        if isinstance(sep_token_id, int) and sep_token_id >= 0:
            eos_token_ids.append(sep_token_id)
        generation = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "eos_token_id": eos_token_ids,
        }
        error_box: list[BaseException] = []

        def generate() -> None:
            try:
                with self.torch.inference_mode():
                    self.model.generate(**generation)
            except BaseException as exc:  # thread boundary: re-raise on the caller
                error_box.append(exc)
                streamer.on_finalized_text("", stream_end=True)

        allocated_before = self.torch.mps.current_allocated_memory()
        started = time.perf_counter()
        first_token_at: float | None = None
        output_parts: list[str] = []
        worker = threading.Thread(target=generate, name="reka-edge-generate", daemon=True)
        worker.start()
        for chunk in streamer:
            if chunk and first_token_at is None:
                first_token_at = time.perf_counter()
            output_parts.append(chunk)
        worker.join(timeout=5)
        finished = time.perf_counter()
        if worker.is_alive():
            raise RuntimeError("Generation thread did not terminate")
        if error_box:
            raise RuntimeError(f"Local generation failed: {error_box[0]}") from error_box[0]
        output_text = "".join(output_parts).replace("<sep>", "").strip()
        output_ids = self.processor.tokenizer.encode(output_text, add_special_tokens=False)
        media_evidence = _media_evidence(image_path)
        record = RequestRecord(
            request_id="local-0001",
            case_id="owned-multimodal-card" if image_path else "local-text",
            provider="transformers-mps",
            model=MODEL_ID,
            evidence_type="observed-local",
            status="ok",
            started_offset_ms=0,
            e2e_ms=(finished - started) * 1000,
            ttft_ms=((first_token_at or finished) - started) * 1000,
            input_tokens=int(inputs["input_ids"].shape[-1]),
            output_tokens=len(output_ids),
            token_count_source="model-tokenizer",
            output_text=output_text,
            metadata={
                "device": "mps",
                "dtype": "float16",
                "model_revision": MODEL_REVISION,
                "model_load_seconds": self.load_seconds,
                "mps_allocated_before_bytes": allocated_before,
                "mps_allocated_after_bytes": self.torch.mps.current_allocated_memory(),
                **media_evidence,
                "max_new_tokens": max_new_tokens,
            },
        )
        benchmark = {
            "schema_version": 1,
            "recorded_at_utc": datetime.now(UTC).isoformat(),
            "evidence_type": "observed-local",
            "provider": "transformers-mps",
            "model": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "system": system_probe(),
            "summary": summarize_records(
                [record], elapsed_seconds=max(finished - started, 1e-9), concurrency=1
            ),
            "records": [record.to_dict()],
            "limitations": [
                "One local request proves execution on this machine, not production capacity.",
                "Load time and first-run compilation differ from warm request latency.",
                "This Transformers/MPS path is not the Linux/CUDA vllm-reka serving path.",
            ],
        }
        return benchmark

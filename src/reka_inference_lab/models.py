"""Validated input and evidence records for the inference lab."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def read_json(path: str | Path) -> dict[str, Any]:
    """Read one JSON object and reject non-object roots."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    """Write stable, reviewable JSON."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _positive_number(value: Any, field_name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    number = float(value)
    if allow_zero:
        if number < 0:
            raise ValueError(f"{field_name} must be zero or greater")
    elif number <= 0:
        raise ValueError(f"{field_name} must be greater than zero")
    return number


@dataclass(frozen=True)
class Workload:
    """Buyer workload assumptions used for a bounded capacity estimate."""

    name: str
    modality: str
    steady_requests_per_second: float
    peak_multiplier: float
    average_input_tokens: int
    average_output_tokens: int
    monthly_requests: int
    target_p95_ms: float
    target_utilization: float
    video_minutes_indexed_per_month: float = 0.0
    video_searches_per_month: int = 0
    privacy_requirement: str = "standard"
    deployment_preference: str = "undecided"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> Workload:
        name = str(data.get("name", "")).strip()
        if not name:
            raise ValueError("name is required")
        modality = str(data.get("modality", "")).strip().lower()
        if modality not in {"text", "image", "video", "audio", "multimodal"}:
            raise ValueError("modality must be text, image, video, audio, or multimodal")
        target_utilization = _positive_number(data.get("target_utilization"), "target_utilization")
        if target_utilization >= 1:
            raise ValueError("target_utilization must be below 1.0")
        return cls(
            name=name,
            modality=modality,
            steady_requests_per_second=_positive_number(
                data.get("steady_requests_per_second"), "steady_requests_per_second"
            ),
            peak_multiplier=_positive_number(data.get("peak_multiplier"), "peak_multiplier"),
            average_input_tokens=int(
                _positive_number(data.get("average_input_tokens"), "average_input_tokens")
            ),
            average_output_tokens=int(
                _positive_number(data.get("average_output_tokens"), "average_output_tokens")
            ),
            monthly_requests=int(
                _positive_number(data.get("monthly_requests"), "monthly_requests")
            ),
            target_p95_ms=_positive_number(data.get("target_p95_ms"), "target_p95_ms"),
            target_utilization=target_utilization,
            video_minutes_indexed_per_month=_positive_number(
                data.get("video_minutes_indexed_per_month", 0),
                "video_minutes_indexed_per_month",
                allow_zero=True,
            ),
            video_searches_per_month=int(
                _positive_number(
                    data.get("video_searches_per_month", 0),
                    "video_searches_per_month",
                    allow_zero=True,
                )
            ),
            privacy_requirement=str(data.get("privacy_requirement", "standard")).strip(),
            deployment_preference=str(data.get("deployment_preference", "undecided")).strip(),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Pricing:
    """Dated public unit prices; private-capacity quotes remain outside this file."""

    name: str
    checked_at: str
    source_url: str
    input_per_million_tokens: float
    output_per_million_tokens: float
    image_each: float = 0.0
    video_per_minute: float = 0.0
    audio_per_minute: float = 0.0
    video_index_per_minute: float = 0.0
    video_search_each: float = 0.0
    notes: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> Pricing:
        required_strings = ("name", "checked_at", "source_url")
        values = {key: str(data.get(key, "")).strip() for key in required_strings}
        missing = [key for key, value in values.items() if not value]
        if missing:
            raise ValueError(f"Missing pricing fields: {', '.join(missing)}")
        notes_data = data.get("notes", [])
        if not isinstance(notes_data, list) or not all(
            isinstance(item, str) for item in notes_data
        ):
            raise ValueError("notes must be a list of strings")

        def money(key: str) -> float:
            return _positive_number(data.get(key, 0), key, allow_zero=True)

        return cls(
            **values,
            input_per_million_tokens=money("input_per_million_tokens"),
            output_per_million_tokens=money("output_per_million_tokens"),
            image_each=money("image_each"),
            video_per_minute=money("video_per_minute"),
            audio_per_minute=money("audio_per_minute"),
            video_index_per_minute=money("video_index_per_minute"),
            video_search_each=money("video_search_each"),
            notes=tuple(notes_data),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["notes"] = list(self.notes)
        return data


@dataclass(frozen=True)
class PromptCase:
    """One owned or user-supplied evaluation request."""

    case_id: str
    prompt: str
    modality: str = "text"
    media_url: str | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> PromptCase:
        case_id = str(data.get("case_id", "")).strip()
        prompt = str(data.get("prompt", "")).strip()
        modality = str(data.get("modality", "text")).strip().lower()
        if not case_id or not prompt:
            raise ValueError("Each prompt needs case_id and prompt")
        if modality not in {"text", "image", "video", "audio"}:
            raise ValueError(f"Unsupported prompt modality: {modality}")
        media_url = data.get("media_url")
        if media_url is not None:
            media_url = str(media_url).strip() or None
        if modality != "text" and media_url is None:
            raise ValueError(f"{case_id} requires media_url for {modality}")
        return cls(case_id=case_id, prompt=prompt, modality=modality, media_url=media_url)


@dataclass
class RequestRecord:
    """One raw request result. Summary metrics are always derived from these records."""

    request_id: str
    case_id: str
    provider: str
    model: str
    evidence_type: str
    status: str
    started_offset_ms: float
    e2e_ms: float
    ttft_ms: float | None
    input_tokens: int | None
    output_tokens: int | None
    token_count_source: str
    output_text: str = ""
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

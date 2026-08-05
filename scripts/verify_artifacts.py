#!/usr/bin/env python3
"""Verify the documented demo bundle from the repository root."""

from __future__ import annotations

import json
from pathlib import Path

from reka_inference_lab.artifacts import verify_bundle


def main() -> int:
    result = verify_bundle(Path("artifacts/demo"))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

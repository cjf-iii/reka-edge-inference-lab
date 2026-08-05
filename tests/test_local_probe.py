from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from reka_inference_lab.local_edge import MODEL_REVISION, _media_evidence, system_probe


class LocalProbeTests(unittest.TestCase):
    def test_probe_is_privacy_reduced_and_pinned(self) -> None:
        probe = system_probe()
        self.assertEqual(probe["model_revision"], MODEL_REVISION)
        self.assertIn("Transformers", probe["boundary"])
        self.assertNotIn("serial", " ".join(probe).lower())

    def test_media_evidence_is_relative_and_hashed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            media = root / "owned.png"
            media.write_bytes(b"owned evaluation bytes")
            previous = Path.cwd()
            os.chdir(root)
            try:
                evidence = _media_evidence(media)
            finally:
                os.chdir(previous)
        self.assertEqual(evidence["image_path"], "owned.png")
        self.assertEqual(evidence["image_path_scope"], "working-directory-relative")
        self.assertEqual(
            evidence["image_sha256"],
            "0ec837d13a20db806a7f45bb8cc017211cac15c4bb71b9a885221f6199a1595c",
        )


if __name__ == "__main__":
    unittest.main()

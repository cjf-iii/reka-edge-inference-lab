from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from reka_inference_lab.cli import build_parser, main


class CliTests(unittest.TestCase):
    def test_demo_runs_end_to_end(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            result = main(
                [
                    "demo",
                    "--prompts",
                    str(root / "examples/prompts.json"),
                    "--workload",
                    str(root / "examples/workload.json"),
                    "--pricing",
                    str(root / "examples/reka_public_pricing_2026-08-04.json"),
                    "--output",
                    temporary,
                    "--requests",
                    "6",
                    "--concurrency",
                    "2",
                ]
            )
            self.assertEqual(result, 0)
            self.assertTrue((Path(temporary) / "decision_report.html").is_file())

    def test_local_requires_explicit_license_acceptance(self) -> None:
        result = main(["local"])
        self.assertEqual(result, 1)

    def test_sweep_levels_parse_as_distinct_integers(self) -> None:
        args = build_parser().parse_args(["sweep", "--levels", "1,3,6"])
        self.assertEqual(args.levels, [1, 3, 6])


if __name__ == "__main__":
    unittest.main()

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detectors.python_detector import detect


class PythonDetectorTests(unittest.TestCase):
    def test_detects_undefined_name(self):
        findings = detect("sample.py", "print(total)\n")

        undefined = [
            finding
            for finding in findings
            if finding["rule"] == "pyflakes_undefinedname"
        ]
        self.assertEqual(len(undefined), 1)
        self.assertEqual(undefined[0]["line_start"], 1)
        self.assertIn("total", undefined[0]["error"])

    def test_detects_literal_division_by_zero(self):
        findings = detect("sample.py", "result = 10 / 0\n")

        division_errors = [
            finding
            for finding in findings
            if finding["rule"] == "division_by_zero"
        ]
        self.assertEqual(len(division_errors), 1)
        self.assertEqual(division_errors[0]["line_start"], 1)
        self.assertEqual(
            division_errors[0]["current_code"],
            "result = 10 / 0",
        )

    def test_separates_undefined_name_and_division_errors(self):
        findings = detect(
            "sample.py",
            "print(total)\nresult = 10 / 0\n",
        )

        detected = {
            finding["rule"]: finding["line_start"]
            for finding in findings
            if finding["rule"] in {
                "pyflakes_undefinedname",
                "division_by_zero",
            }
        }
        self.assertEqual(
            detected,
            {
                "pyflakes_undefinedname": 1,
                "division_by_zero": 2,
            },
        )


if __name__ == "__main__":
    unittest.main()

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_client import _deterministic_repair


class SyntaxRepairTests(unittest.TestCase):
    def test_deterministic_repair_closes_unclosed_bracket(self):
        finding = {
            "rule": "syntax_error",
            "error": "Unclosed '['",
            "code_chunk": "numbers = [1, 2",
        }

        replacement = _deterministic_repair(finding)

        self.assertEqual(replacement, "numbers = [1, 2]")
        compile(replacement, "<corrected-test>", "exec")

    def test_deterministic_repair_closes_unterminated_string(self):
        finding = {
            "rule": "syntax_error",
            "error": "unterminated string literal",
            "code_chunk": 'name = "Munna',
        }

        replacement = _deterministic_repair(finding)

        self.assertEqual(replacement, 'name = "Munna"')
        compile(replacement, "<corrected-test>", "exec")


if __name__ == "__main__":
    unittest.main()

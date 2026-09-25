import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detectors.python_detector import detect
from llm_client import get_fallback_report


TEST_CASES = [
    """def calculate_sum(a, b)
    return a + b

def greet(name:
    print("Hello", name)

for i in range(5)
    print(i)

if 10 > 5
    print("Ten is greater")

numbers = [1, 2, 3, 4
print(numbers)

def multiply(x, y):
    return x * y))

class Student
    def __init__(self, name):
        self.name = name

try:
    x = 10 / 0
except ZeroDivisionError
    print("Cannot divide by zero")

while True
    break

print("Program finished"
""",
    """def add(a b):
    return a + b

if True print("Hello")

name = "Munna

numbers = [10, 20, 30

break

return 100

continue

def multiply(x, x):
    return x * x

print(a=10, 20)

from math import
""",
]


class SyntaxRepairTests(unittest.TestCase):
    def test_both_10_error_examples_have_10_findings_and_replacements(self):
        for source in TEST_CASES:
            with self.subTest(source=source[:30]):
                findings = detect("s_error.py", source)
                self.assertEqual(len(findings), 10)
                reports = [get_fallback_report(f) for f in findings]
                self.assertTrue(all(r.get("replacement_code") for r in reports))

                corrected_lines = source.splitlines()
                for finding, report in zip(findings, reports):
                    line_index = finding["line_start"] - 1
                    corrected_lines[line_index] = report["replacement_code"]

                corrected = "\n".join(corrected_lines)
                compile(corrected, "<corrected-test>", "exec")


if __name__ == "__main__":
    unittest.main()

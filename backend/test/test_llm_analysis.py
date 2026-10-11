import sys
import unittest
import os
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import llm_client
from llm_client import (
    LLM_UNAVAILABLE_MESSAGE,
    _build_user_message,
    analyze_finding,
)


class LlmAnalysisTests(unittest.TestCase):
    def setUp(self):
        provider_request = patch(
            "llm_client.requests.post",
            side_effect=AssertionError(
                "Provider HTTP requests must be mocked in unit tests."
            ),
        )
        provider_request.start()
        self.addCleanup(provider_request.stop)

    def test_groq_loader_reads_five_numbered_keys(self):
        keys = {
            f"GROQ_API_KEY_{number}": f"groq-test-{number}"
            for number in range(1, 6)
        }
        with patch.dict(os.environ, keys, clear=False):
            self.assertEqual(
                llm_client._get_groq_api_keys(),
                [f"groq-test-{number}" for number in range(1, 6)],
            )

    def test_gemini_loader_does_not_truncate_configured_keys(self):
        keys = {
            f"GEMINI_API_KEY_{number}": f"gemini-test-{number}"
            for number in range(1, 12)
        }
        keys["GEMINI_API_KEY"] = "gemini-legacy-test"
        with patch.dict(os.environ, keys, clear=True):
            self.assertEqual(
                llm_client._get_api_keys(),
                [
                    *(f"gemini-test-{number}" for number in range(1, 12)),
                    "gemini-legacy-test",
                ],
            )

    def test_each_finding_gets_its_own_code_context(self):
        first = _build_user_message(
            {
                "file": "first.py",
                "rule": "possible_division_by_zero",
                "error": "denominator may be zero",
                "current_code": "total / count",
                "code_chunk": "def average(total, count):\n    return total / count",
            },
            [],
        )
        second = _build_user_message(
            {
                "file": "second.py",
                "rule": "mutable_default_arg",
                "error": "default list is shared",
                "current_code": "items=[]",
                "code_chunk": "def add(item, items=[]):\n    items.append(item)",
            },
            [],
        )

        self.assertIn("total / count", first)
        self.assertIn("first.py", first)
        self.assertNotIn("items.append(item)", first)
        self.assertIn("items.append(item)", second)
        self.assertIn("second.py", second)
        self.assertNotIn("total / count", second)

    def test_missing_api_key_does_not_invent_an_explanation(self):
        finding = {
            "rule": "possible_division_by_zero",
            "bug_type": "Runtime Error",
            "error": "denominator may be zero",
            "cause": "count could be zero",
            "current_code": "return total / count",
            "code_chunk": "return total / count",
        }

        with (
            patch("llm_client._get_provider_credentials", return_value=[]),
        ):
            result = analyze_finding(finding, [])

        self.assertEqual(result["analysis_source"], "unavailable")
        self.assertTrue(result["llm_unavailable"])
        self.assertEqual(result["cause"], "")
        self.assertEqual(result["solution"], "")
        self.assertIsNone(result["replacement_code"])
        self.assertEqual(result["llm_unavailable_notice"], LLM_UNAVAILABLE_MESSAGE)

    def test_incomplete_llm_response_does_not_get_filled_with_fallback_text(self):
        finding = {
            "rule": "undefined_name",
            "bug_type": "Runtime Error",
            "error": "undefined name 'total'",
            "current_code": "print(total)",
            "code_chunk": "print(total)",
        }
        response = type(
            "Response",
            (),
            {
                "status_code": 200,
                "json": lambda self: {
                    "candidates": [{
                        "content": {
                            "parts": [{"text": '{"cause":"","solution":""}'}]
                        }
                    }]
                },
            },
        )()

        with (
            patch(
                "llm_client._get_provider_credentials",
                return_value=[("gemini", 1, "test-key")],
            ),
            patch("llm_client.requests.post", return_value=response),
        ):
            result = analyze_finding(finding, [])

        self.assertTrue(result["llm_unavailable"])
        self.assertEqual(result["cause"], "")
        self.assertEqual(result["solution"], "")
        self.assertEqual(result["llm_unavailable_notice"], LLM_UNAVAILABLE_MESSAGE)

    def test_usage_limit_returns_the_required_unavailable_message(self):
        finding = {
            "error": "undefined name 'total'",
            "bug_type": "Runtime Error",
        }
        response = type("Response", (), {"status_code": 429})()

        with (
            patch(
                "llm_client._get_provider_credentials",
                return_value=[("gemini", 1, "test-key")],
            ),
            patch("llm_client.requests.post", return_value=response),
        ):
            result = analyze_finding(finding, [])

        self.assertTrue(result["llm_unavailable"])
        self.assertTrue(result["rate_limited"])
        self.assertEqual(result["cause"], "")
        self.assertEqual(result["solution"], "")
        self.assertEqual(result["llm_unavailable_notice"], LLM_UNAVAILABLE_MESSAGE)

    def test_tries_every_gemini_key_before_reporting_quota_unavailable(self):
        limited_response = type("Response", (), {"status_code": 429})()

        with (
            patch(
                "llm_client._get_provider_credentials",
                return_value=[
                    ("gemini", 1, "first-key"),
                    ("gemini", 2, "second-key"),
                    ("gemini", 3, "third-key"),
                ],
            ),
            patch(
                "llm_client.requests.post",
                return_value=limited_response,
            ) as post,
        ):
            result = analyze_finding(
                {"error": "test finding", "bug_type": "Other"},
                [],
            )

        self.assertEqual(post.call_count, 3)
        self.assertTrue(result["llm_unavailable"])
        self.assertTrue(result["rate_limited"])
        self.assertEqual(result["cause"], "")
        self.assertEqual(result["solution"], "")

    def test_rate_limited_gemini_key_rotates_to_the_next_key(self):
        limited_response = type("Response", (), {"status_code": 429})()
        successful_response = type(
            "Response",
            (),
            {
                "status_code": 200,
                "json": lambda self: {
                    "candidates": [{
                        "content": {
                            "parts": [{
                                "text": (
                                    '{"error":"Undefined name total",'
                                    '"bug_type":"Runtime Error",'
                                    '"cause":"The code uses total before it is defined.",'
                                    '"solution":"Define total before this print statement.",'
                                    '"replacement_code":"total = 0\\nprint(total)",'
                                    '"confidence":90}'
                                )
                            }]
                        }
                    }]
                },
            },
        )()

        with (
            patch(
                "llm_client._get_provider_credentials",
                return_value=[
                    ("gemini", 1, "first-key"),
                    ("gemini", 2, "second-key"),
                ],
            ),
            patch(
                "llm_client.requests.post",
                side_effect=[limited_response, successful_response],
            ) as post,
        ):
            result = analyze_finding(
                {
                    "error": "undefined name 'total'",
                    "bug_type": "Runtime Error",
                    "current_code": "print(total)",
                    "code_chunk": "print(total)",
                },
                [],
            )

        self.assertEqual(post.call_count, 2)
        self.assertEqual(result["analysis_source"], "gemini")
        self.assertEqual(result["cause"], "The code uses total before it is defined.")
        self.assertEqual(result["solution"], "Define total before this print statement.")
        self.assertFalse(result.get("llm_unavailable", False))

    def test_rotating_provider_pool_reaches_fifth_groq_key(self):
        successful_groq_response = (
            '{"error":"Undefined name total",'
            '"bug_type":"Runtime Error",'
            '"cause":"The code prints total before assigning it a value.",'
            '"solution":"Create total before calling print(total).",'
            '"replacement_code":"total = 0\\nprint(total)",'
            '"confidence":90}'
        )

        with (
            patch(
                "llm_client._get_provider_credentials",
                return_value=[
                    ("gemini", 1, "gemini-test"),
                    ("groq", 1, "groq-test-1"),
                    ("gemini", 2, "gemini-test-2"),
                    ("groq", 2, "groq-test-2"),
                    ("gemini", 3, "gemini-test-3"),
                    ("groq", 3, "groq-test-3"),
                    ("gemini", 4, "gemini-test-4"),
                    ("groq", 4, "groq-test-4"),
                    ("gemini", 5, "gemini-test-5"),
                    ("groq", 5, "groq-test-5"),
                ],
            ),
            patch(
                "llm_client.requests.post",
                return_value=type("Response", (), {"status_code": 429})(),
            ),
            patch(
                "llm_client._groq_request",
                side_effect=[
                    RuntimeError("rate limited"),
                    RuntimeError("rate limited"),
                    RuntimeError("rate limited"),
                    RuntimeError("rate limited"),
                    successful_groq_response,
                ],
            ) as groq_request,
        ):
            result = analyze_finding(
                {
                    "error": "undefined name 'total'",
                    "bug_type": "Runtime Error",
                    "current_code": "print(total)",
                    "code_chunk": "print(total)",
                },
                [],
            )

        self.assertEqual(groq_request.call_count, 5)
        self.assertEqual(result["analysis_source"], "groq")
        self.assertFalse(result.get("llm_unavailable", False))
        self.assertIn("prints total", result["cause"])
        self.assertIn("Create total", result["solution"])

    def test_provider_pool_rotates_starting_key_between_findings(self):
        credentials = [
            ("gemini", 1, "gemini-test-1"),
            ("groq", 1, "groq-test-1"),
            ("gemini", 2, "gemini-test-2"),
            ("groq", 2, "groq-test-2"),
        ]

        with (
            patch(
                "llm_client._get_api_keys",
                return_value=["gemini-test-1", "gemini-test-2"],
            ),
            patch(
                "llm_client._get_groq_api_keys",
                return_value=["groq-test-1", "groq-test-2"],
            ),
            patch("llm_client._PROVIDER_ROTATION_INDEX", 0),
        ):
            first = llm_client._get_provider_credentials()
            second = llm_client._get_provider_credentials()

        self.assertEqual(first, credentials)
        self.assertEqual(
            second,
            credentials[1:] + credentials[:1],
        )


if __name__ == "__main__":
    unittest.main()

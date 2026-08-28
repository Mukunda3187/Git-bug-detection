"""
Wraps the call to the LLM (Google Gemini). Give it:
  - a raw candidate finding from a detector (detectors/python_detector.py)
  - the retrieved similar historical bugs from the RAG step
and it returns a fully-formed structured bug report as a dict, matching
models.BugReport (minus the fields the caller already knows, like id/file).

Uses Gemini's REST API directly via `requests` (already a dependency) -
no extra SDK to install or version-pin.

If no GEMINI_API_KEY is set, falls back to a transparent rule-based
formatter so the app still runs end-to-end for a demo - it clearly does
NOT pretend to be the LLM's reasoning, it just formats what the detector
already found.
"""
import json
import math
import os

import requests

GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"


def _parse_rate_limit_info(resp):
    """
    Reads Google's 429 error body to figure out how long to wait and whether
    this is a short per-minute limit or the daily free-tier cap. Returns
    (retry_seconds_or_None, is_daily_limit).
    """
    try:
        data = resp.json()
    except (ValueError, json.JSONDecodeError):
        return None, False

    error = data.get("error", {})
    details = error.get("details", [])
    retry_seconds = None
    is_daily = False

    for d in details:
        type_str = d.get("@type", "")
        if type_str.endswith("RetryInfo"):
            delay = d.get("retryDelay", "")
            try:
                retry_seconds = float(str(delay).rstrip("s"))
            except ValueError:
                pass
        if type_str.endswith("QuotaFailure"):
            for v in d.get("violations", []):
                combined = f"{v.get('quotaId', '')} {v.get('quotaMetric', '')}"
                if "PerDay" in combined or "per_day" in combined:
                    is_daily = True

    return retry_seconds, is_daily


def _build_rate_limit_message(retry_seconds, is_daily):
    """Turns the parsed rate-limit details into one plain-English sentence."""
    if is_daily:
        if retry_seconds:
            minutes = max(1, math.ceil(retry_seconds / 60))
            return f"The free AI usage limit for today has been reached. Try again in about {minutes} minute(s), or after Google's daily reset (midnight Pacific Time, US)."
        return "The free AI usage limit for today has been reached. It resets at midnight Pacific Time (US) - please try again after that."

    if retry_seconds:
        if retry_seconds < 60:
            wait = int(retry_seconds) + 5  # small buffer
            return f"The AI is temporarily busy. Try again in about {wait} seconds."
        minutes = max(1, math.ceil(retry_seconds / 60))
        return f"The AI is temporarily busy. Try again in about {minutes} minute(s)."

    return "The AI usage limit has been reached for now. Please try again in a minute or two."

SYSTEM_PROMPT = """You are an expert code reviewer helping explain bugs to developers. Your goal is to provide clear, 
actionable solutions that can be immediately applied.

For each reported issue, provide a response with:
1. **cause**: Explain EXACTLY why this code is problematic in 2-3 sentences, no jargon.
2. **solution_type**: Choose from "replace", "add", "remove", or "create_file" - pick the most direct fix.
3. **replacement_code**: For "replace" solutions, provide the EXACT corrected code that can be directly substituted.
   - Keep variable names and structure identical
   - Only modify the problematic part
   - Ensure it's production-ready and handles edge cases
4. **action**: One specific sentence describing what to do (e.g., "Replace line 42 with the corrected code" or "Add this try-except block after the variable assignment").
5. **explanation**: Provide a brief technical explanation of why the fix works.

CRITICAL RULES:
- If solution_type is "replace", replacement_code MUST NEVER be null or empty. Always provide working code.
- If you cannot provide a concrete fix, use solution_type "add" or set insufficient_evidence to true.
- Do not paraphrase or restructure unrelated code.
- Provide solutions that are immediately applicable to the codebase.
- Consider edge cases and error handling in your fixes.

Reply with ONLY a JSON object, no markdown fences or extra text:
{
  "error": string,
  "bug_type": one of ["Runtime Error","Logic Error","Syntax Error","Type Error","Dependency Error","Security Issue","Performance Issue","API Error","Unnecessary Code","Other"],
  "cause": string,
  "why_occurs": string,
  "solution_type": one of ["replace","add","remove","create_file"],
  "solution_intro": string,
  "replacement_code": string or null,
  "add_location": string or null,
  "new_file_path": string or null,
  "action": string,
  "explanation": string,
  "insufficient_evidence": boolean
}"""


def _build_user_message(finding: dict, retrieved: list) -> str:
    retrieved_block = "\n\n".join(
        f"- Dataset: {r['record'].get('dataset_source')}\n"
        f"  Bug type: {r['record'].get('bug_type')}\n"
        f"  Description: {r['record'].get('bug_description')}\n"
        f"  Solution: {r['record'].get('solution', 'N/A')}\n"
        f"  Similarity: {round(r['similarity'] * 100)}%"
        for r in retrieved
    ) or "(no similar historical bugs found)"

    return f"""**Candidate Issue from Static Analysis:**

**File:** {finding.get('file')}
**Function:** {finding.get('function') or 'N/A'}
**Lines:** {finding.get('line_start')}-{finding.get('line_end')}
**Rule:** {finding.get('rule')}
**Initial Assessment:** {finding.get('error')} ({finding.get('bug_type')})
**Detector's Note:** {finding.get('cause')}

**Current Code:**
```
{finding.get('current_code')}
```

**Historical Context (Similar Bugs from Knowledge Base):**
{retrieved_block}

**Your Task:**
Analyze this candidate and provide a complete, production-ready fix. Ensure replacement_code is never empty for "replace" solutions."""


def _fallback_report(finding: dict) -> dict:
    """Used when no LLM reasoning is available - either no API key is configured, or the
    live LLM call failed/timed out. Kept in plain, non-technical English since this is
    shown directly to the end user - technical failure details are logged to the server
    console instead (see analyze_finding), never shown in the UI.

    Every rule every detector can produce gets a real, specific template here -
    no rule should ever fall through to a generic "review and fix it" message.
    """
    rule = finding.get("rule", "")
    bug_type = finding.get("bug_type", "Other")
    current_code = finding.get("current_code", "") or ""
    cause_from_detector = finding.get("cause", "")

    # Defaults - overridden below per rule. solution_type "replace" needs
    # replacement_code; "remove" and "add" generally don't need it filled in
    # for a deterministic fallback since there's nothing left to guess.
    solution_type = "replace"
    solution_intro = ""
    action = ""
    replacement_code = None
    add_location = None
    cause = cause_from_detector

    if rule == "possibly_unused_function":
        solution_type = "remove"
        solution_intro = "Remove this code from the file."
        action = "Delete this unused function."

    elif rule == "bare_except":
        solution_type = "replace"
        solution_intro = "Replace the given code with the new code shown below."
        replacement_code = current_code.replace("except:", "except Exception as e:", 1)
        action = "Replace the bare 'except:' with 'except Exception as e:' so real errors aren't silently hidden."

    elif rule == "mutable_default_arg":
        solution_type = "replace"
        solution_intro = "Replace the given code with the new code shown below."
        action = "Change the default value to None, then create a new list/dict inside the function body."

    elif rule == "eq_none":
        solution_type = "replace"
        solution_intro = "Replace the given code with the new code shown below."
        replacement_code = current_code.replace("== None", "is None").replace("!= None", "is not None")
        action = "Use 'is None' / 'is not None' instead of '==' / '!=' when comparing to None."

    elif rule == "possible_division_by_zero":
        solution_type = "add"
        solution_intro = "Add a check for zero before this line, as shown below."
        add_location = "Add this check on the line right before the division."
        replacement_code = "if denominator != 0:  # replace 'denominator' with your actual variable name"
        action = "Add a zero-check before dividing, so the program doesn't crash if the value is zero."

    elif rule == "syntax_error":
        solution_type = "replace"
        solution_intro = "There is a Python syntax error on this line that needs to be fixed by hand."
        cause = f"Python's own parser could not read this code. The exact reason it gave was: \"{cause_from_detector}\"."
        action = f"Look closely at this line and fix the syntax issue Python reported: {cause_from_detector}."

    elif rule in ("unclosed_bracket", "mismatched_bracket", "unexpected_closing_bracket"):
        solution_type = "replace"
        solution_intro = "There is a bracket that doesn't match up correctly - fix it by hand at the location shown."
        action = "Check every '{', '(' and '[' near this line and make sure each one has a matching closing bracket in the right order."

    elif rule == "unterminated_string":
        solution_type = "replace"
        solution_intro = "A text string on this line is missing its closing quote."
        action = "Add the missing closing quote (matching the one that opened the string) at the end of the text."

    elif rule == "unterminated_template_literal":
        solution_type = "replace"
        solution_intro = "A template string (using backticks) on this line is missing its closing backtick."
        action = "Add the missing closing backtick ( ` ) to complete the template string."

    elif rule == "unterminated_comment":
        solution_type = "replace"
        solution_intro = "A block comment was opened here but never closed."
        action = "Add the missing */ to close this comment block."

    elif rule == "loose_equality":
        solution_type = "replace"
        solution_intro = "Replace the given code with the new code shown below."
        if "!=" in current_code:
            replacement_code = current_code.replace("!=", "!==")
            action = "Change '!=' to '!==' so values are compared without unexpected type conversion."
        else:
            replacement_code = current_code.replace("==", "===")
            action = "Change '==' to '===' so values are compared without unexpected type conversion."

    elif rule == "var_declaration":
        solution_type = "replace"
        solution_intro = "Replace the given code with the new code shown below."
        replacement_code = current_code.replace("var ", "let ", 1)
        action = "Change 'var' to 'let' (or 'const' if this value is never reassigned)."

    elif rule == "empty_catch_block":
        solution_type = "replace"
        solution_intro = "Replace the given code with the new code shown below."
        action = "Add at least a console.error(err) inside the catch block so failures aren't silently swallowed."

    elif rule == "leftover_console_statement":
        solution_type = "remove"
        solution_intro = "Remove this code from the file."
        action = "Delete this console.log/debug statement before shipping."

    elif rule == "leftover_debugger_statement":
        solution_type = "remove"
        solution_intro = "Remove this code from the file."
        action = "Delete this 'debugger' statement before shipping."

    else:
        # Should not normally be reached - every known rule is handled above -
        # but keep a safe, honest fallback for any future/unknown rule.
        solution_type = "replace"
        solution_intro = "We could not prepare an automatic fix for this one right now - please look at the code below and fix it yourself."
        action = "Review this code and fix it yourself."

    return {
        "error": finding.get("error", "Possible issue"),
        "bug_type": bug_type,
        "cause": cause or "Something in this code looks like it could cause a problem.",
        "why_occurs": "",
        "solution_type": solution_type,
        "solution_intro": solution_intro,
        "replacement_code": replacement_code,
        "add_location": add_location,
        "new_file_path": None,
        "action": action,
        "explanation": "",
        "insufficient_evidence": True,
    }


def analyze_finding(finding: dict, retrieved: list) -> dict:
    api_keys = []

    # Read multiple Gemini API keys from environment variables.
    for i in range(1, 11):
        key = os.getenv(f"GEMINI_API_KEY_{i}")
        if key and key.strip():
            api_keys.append(key.strip())

    # Keep support for the old single-key variable.
    old_key = os.getenv("GEMINI_API_KEY")
    if old_key and old_key.strip() and old_key.strip() not in api_keys:
        api_keys.append(old_key.strip())

    if not api_keys:
        return _fallback_report(finding)

    payload = {
        "system_instruction": {
            "parts": [{"text": SYSTEM_PROMPT}]
        },
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": _build_user_message(
                            finding,
                            retrieved
                        )
                    }
                ],
            }
        ],
        "generationConfig": {
            "response_mime_type": "application/json",
            "temperature": 0.1,  # Lower for more consistent/reliable outputs
            "maxOutputTokens": 1500,  # Increased for better solutions
            "topP": 0.9,
            "topK": 40,
        },
    }

    last_error = None
    last_429_response = None

    # Try each Gemini API key until one succeeds.
    for key_number, api_key in enumerate(api_keys, start=1):
        try:
            resp = requests.post(
                GEMINI_URL,
                params={"key": api_key},
                json=payload,
                timeout=12,
            )

            if resp.status_code == 200:
                data = resp.json()
                raw_text = data["candidates"][0]["content"]["parts"][0]["text"]

                try:
                    cleaned = (
                        raw_text
                        .strip()
                        .removeprefix("```json")
                        .removeprefix("```")
                        .removesuffix("```")
                        .strip()
                    )

                    result = json.loads(cleaned)
                    
                    # Validate that replacement_code is not empty for "replace" solutions
                    if result.get("solution_type") == "replace" and not result.get("replacement_code"):
                        print(f"[llm_client] Gemini returned empty replacement_code for replace solution. Using fallback.")
                        return _fallback_report(finding)
                    
                    return result

                except (json.JSONDecodeError, ValueError) as e:
                    print(f"[llm_client] Gemini returned non-JSON response: {raw_text[:300]}")
                    print(f"[llm_client] JSON parse error: {e}")
                    return _fallback_report(finding)

            if resp.status_code == 429:
                last_429_response = resp

            # Key failed - try the next key. Log the technical detail server-side
            # only - the person using the app should never see raw HTTP/API errors.
            last_error = f"Gemini key {key_number} returned HTTP {resp.status_code}: {resp.text[:200]}"

        except Exception as e:
            last_error = f"Gemini key {key_number} failed: {e}"

    # All keys failed. Print the real reason to the server logs (visible in Render's
    # Logs tab) so it can still be debugged, but keep the user-facing result simple.
    print(f"[llm_client] All Gemini API keys failed. Last error: {last_error}")

    fallback = _fallback_report(finding)

    # If every key failed specifically because of a rate limit, tell the person
    # clearly (and when they can try again) instead of the generic "fix it
    # yourself" message - this is shown once at the top of the scan results.
    if last_429_response is not None:
        retry_seconds, is_daily = _parse_rate_limit_info(last_429_response)
        fallback["rate_limited"] = True
        fallback["rate_limit_message"] = _build_rate_limit_message(retry_seconds, is_daily)

    return fallback

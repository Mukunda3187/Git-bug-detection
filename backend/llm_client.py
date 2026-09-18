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
import re

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
4. **solution**: ONE clear, complete, specific passage (2-4 sentences) that fully explains the fix -
   what exactly to change or do, in concrete terms tied to this exact code (line numbers, variable
   names, the exact characters involved), not generic advice that could apply to any bug of this
   type. This is the ONLY place the person reads for "what do I do" - do not split this into a vague
   intro plus a separate action elsewhere; say everything they need in this one passage. If the fix
   is a code replacement, describe what changed and why, since the code itself is also shown separately.
5. **explanation**: Provide a brief technical explanation of why the fix works.
6. **confidence**: An integer from 0 to 100 for THIS SPECIFIC finding, reflecting how confident you are that
   (a) this is genuinely a real, correctly-diagnosed problem given the code and context shown, and
   (b) the fix you're proposing actually resolves it correctly and completely.
   Use the full range honestly - a solid, unambiguous fix for a clear-cut issue deserves 90-100.
   A fix you're reasonably sure about but that depends on context you can't fully see deserves 60-85.
   Anything you're genuinely unsure about, where the retrieved historical evidence is thin or the fix
   is a best guess, deserves 30-55. Do not default to a single "safe" number every time - vary it
   honestly based on the actual evidence for this specific finding.

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
  "solution": string,
  "replacement_code": string or null,
  "add_location": string or null,
  "new_file_path": string or null,
  "explanation": string,
  "confidence": integer from 0 to 100,
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


# How confident the FALLBACK path (no live LLM call) can honestly be in
# its own suggested fix, per rule. This varies deliberately - a rule with
# an exact, verified string-substitution fix (like eq_none: "== None" -> "is None")
# deserves a much higher number than one with only generic advice and no
# computed replacement (like empty_catch_block, which just says "add a
# console.error"). This is about confidence in the FIX, not confidence
# that the underlying finding is real - every finding these detectors
# produce is already a 100%-certain fact (a real syntax error, or
# provably unreachable code), independent of this fallback quality score.
FALLBACK_CONFIDENCE_BY_RULE = {
    "possibly_unused_function": 60,
    "bare_except": 85,
    "mutable_default_arg": 55,
    "eq_none": 90,
    "possible_division_by_zero": 50,
    "syntax_error": 70,
    "unclosed_bracket": 65,
    "mismatched_bracket": 65,
    "unexpected_closing_bracket": 65,
    "unterminated_string": 75,
    "unterminated_template_literal": 75,
    "unterminated_comment": 75,
    "loose_equality": 90,
    "var_declaration": 85,
    "empty_catch_block": 55,
    "leftover_console_statement": 90,
    "leftover_debugger_statement": 90,
    "leftover_debug_print": 85,
    "unreachable_code": 95,  # deleting provably-dead code is always a safe, correct fix
}
DEFAULT_FALLBACK_CONFIDENCE = 40

# Pulls the exact character(s) the detector already identified out of its
# "error" string, so the fallback solution can name them specifically
# ("add the missing '}'") instead of falling back to generic advice
# ("check every bracket near this line") when the detector already knows
# precisely which bracket and which problem this is.
UNCLOSED_CHAR_RE = re.compile(r"Unclosed '(.)'")
MISMATCHED_CHARS_RE = re.compile(r"found '(.)', expected '(.)'")
UNEXPECTED_CHAR_RE = re.compile(r"Unexpected '(.)'")
BRACKET_CLOSER = {"(": ")", "[": "]", "{": "}"}

# Matches a single-line-signature mutable default like "bucket=[]", "cfg={}",
# or "seen=set()" - covers the common case the fallback path can fix safely
# without a real parser (multi-line signatures are left alone, see below).
MUTABLE_DEFAULT_RE = re.compile(r"(\w+)\s*=\s*(\[\]|\{\}|set\(\))")

# Matches an empty catch block, with or without a captured error variable,
# e.g. "catch (err) {}", "catch (Exception e) {}", or "catch {}".
EMPTY_CATCH_WITH_PARAM_RE = re.compile(r"catch\s*\(([^)]*)\)\s*\{\s*\}")
EMPTY_CATCH_NO_PARAM_RE = re.compile(r"catch\s*\{\s*\}")

CATCH_LOG_STATEMENT_BY_EXTENSION = {
    ".js": "console.error({var});", ".jsx": "console.error({var});",
    ".ts": "console.error({var});", ".tsx": "console.error({var});",
    ".java": "{var}.printStackTrace();",
    ".cs": "Console.WriteLine({var});",
    ".cpp": "std::cerr << {var}.what() << std::endl;",
    ".php": "error_log({var}->getMessage());",
}


def _fix_mutable_default_arg(current_code: str):
    """
    Mechanically rewrites a one-line function signature with a mutable
    default argument into the safe None-default pattern, e.g.:
        def add_item(item, bucket=[]):        ->  def add_item(item, bucket=None):
            bucket.append(item)                        if bucket is None:
                                                            bucket = []
                                                        bucket.append(item)
    Returns None (never a half-applied guess) when the default can't be
    confidently located, so the caller can fall back to plain-text advice
    instead of claiming code is shown when it isn't.
    """
    lines = current_code.splitlines()
    if not lines:
        return None

    match = MUTABLE_DEFAULT_RE.search(lines[0])
    if not match:
        return None

    param_name, literal = match.group(1), match.group(2)
    default_value = {"[]": "[]", "{}": "{}", "set()": "set()"}[literal]
    new_signature = lines[0][:match.start()] + f"{param_name}=None" + lines[0][match.end():]

    body_lines = lines[1:]
    indent = "    "
    if body_lines:
        stripped = body_lines[0].lstrip()
        indent = body_lines[0][:len(body_lines[0]) - len(stripped)] or indent

    guard = [f"{indent}if {param_name} is None:", f"{indent}    {param_name} = {default_value}"]
    return "\n".join([new_signature] + guard + body_lines)


def _fix_empty_catch(current_code: str, file_path: str):
    """
    Mechanically rewrites a one-line empty catch block into one that logs
    the error, using a log statement appropriate to the file's language
    (inferred from its extension) and re-using whatever variable name the
    catch clause already captured. Returns None if no empty catch pattern
    is found, rather than guessing.
    """
    ext = os.path.splitext(file_path or "")[1]
    log_template = CATCH_LOG_STATEMENT_BY_EXTENSION.get(ext)
    if not log_template:
        return None

    match = EMPTY_CATCH_WITH_PARAM_RE.search(current_code)
    if match:
        param_str = match.group(1).strip()
        var_match = re.search(r"(\$?\w+)\s*$", param_str)
        var_name = var_match.group(1) if var_match else "e"
        log_stmt = log_template.format(var=var_name)
        return current_code[:match.start()] + f"catch ({param_str}) {{ {log_stmt} }}" + current_code[match.end():]

    match = EMPTY_CATCH_NO_PARAM_RE.search(current_code)
    if match:
        var_name = "err" if ext in (".js", ".jsx", ".ts", ".tsx") else "e"
        log_stmt = log_template.format(var=var_name)
        return current_code[:match.start()] + f"catch ({var_name}) {{ {log_stmt} }}" + current_code[match.end():]

    return None


def _fallback_report(finding: dict) -> dict:
    """Used when no LLM reasoning is available - either no API key is configured, or the
    live LLM call failed/timed out. Kept in plain, non-technical English since this is
    shown directly to the end user - technical failure details are logged to the server
    console instead (see analyze_finding), never shown in the UI.

    Every rule every detector can produce gets a real, specific template here - no rule
    should ever fall through to a generic "review and fix it" message. Each one sets a
    single `solution` string that stands completely on its own (what's wrong here,
    specifically, and exactly what to do about it) rather than the old split between a
    generic intro sentence and a separate action sentence - for the three bracket rules
    in particular, that generic intro used to be the same wording no matter which
    specific bracket or which specific problem the detector had already identified, even
    though that exact detail (which character, unclosed vs mismatched vs unexpected) was
    sitting right there in the detector's own `error` string. This pulls it out and names
    it directly instead of leaving the person to re-derive it from the code themselves.
    """
    rule = finding.get("rule", "")
    bug_type = finding.get("bug_type", "Other")
    current_code = finding.get("current_code", "") or ""
    error_text = finding.get("error", "") or ""
    cause_from_detector = finding.get("cause", "")

    # Defaults - overridden below per rule. solution_type "replace" needs
    # replacement_code; "remove" and "add" generally don't need it filled in
    # for a deterministic fallback since there's nothing left to guess.
    solution_type = "replace"
    solution = ""
    replacement_code = None
    add_location = None
    cause = cause_from_detector

    if rule == "possibly_unused_function":
        solution_type = "remove"
        solution = "Delete this function. It isn't called anywhere else in this file, so removing it has no effect on behavior - if it turns out to be used from another file this detector can't see, undo the deletion instead of guessing."

    elif rule == "bare_except":
        solution_type = "replace"
        replacement_code = current_code.replace("except:", "except Exception as e:", 1)
        solution = "Replace the bare 'except:' with 'except Exception as e:', as shown below. A bare except also catches things like KeyboardInterrupt and SystemExit, which should almost never be silently swallowed - naming Exception avoids that while still catching ordinary errors."

    elif rule == "mutable_default_arg":
        replacement_code = _fix_mutable_default_arg(current_code)
        if replacement_code:
            solution_type = "replace"
            solution = "Replace this signature with the version below: change the default to None, then create a fresh list/dict/set inside the function body on first use. The current version reuses the SAME object across every call, so items added in one call silently show up in the next."
        else:
            # Couldn't mechanically locate the mutable default in the captured
            # snippet (e.g. a multi-line signature) - never claim code is
            # "shown below" and then show nothing.
            solution_type = "add"
            solution = "Change the default value to None, then add 'if <param> is None: <param> = []' (or {} / set(), matching whatever the original default was) as the first line inside the function body. This function's mutable default is currently shared across every call to it, which usually isn't intended."

    elif rule == "eq_none":
        solution_type = "replace"
        replacement_code = current_code.replace("== None", "is None").replace("!= None", "is not None")
        solution = "Replace '==' / '!=' with 'is' / 'is not' when comparing to None, as shown below. 'is' checks identity directly and can't be fooled by a custom __eq__ method, which is why it's the correct way to check for None in Python."

    elif rule == "possible_division_by_zero":
        solution_type = "add"
        add_location = "Add this check on the line right before the division."
        replacement_code = "if denominator != 0:  # replace 'denominator' with your actual variable name"
        solution = "Add a check that the denominator isn't zero before this line runs (see the line to add below), and decide what should happen when it is - skip the calculation, return a default value, or raise a clear error instead of letting the program crash with a ZeroDivisionError."

    elif rule == "syntax_error":
        solution_type = "replace"
        cause = f"Python's own parser could not read this code. The exact reason it gave was: \"{cause_from_detector}\"."
        solution = f"Edit this line to fix the specific problem Python's parser reported: {cause_from_detector}. The file won't run at all until this is fixed, since Python can't even finish reading it - after editing, re-run the file (or 'python -m py_compile <file>') to confirm it now parses cleanly."

    elif rule == "unclosed_bracket":
        solution_type = "replace"
        m = UNCLOSED_CHAR_RE.search(error_text)
        opener = m.group(1) if m else "{"
        closer = BRACKET_CLOSER.get(opener, "}")
        solution = (
            f"A '{opener}' was opened here but is never closed anywhere in the rest of the file. "
            f"Add the missing '{closer}' at the point where this block, function call, or expression "
            f"is meant to end - if you're not sure exactly where, work outward from this line counting "
            f"'{opener}' and '{closer}' until you find the spot where one is missing."
        )

    elif rule == "mismatched_bracket":
        solution_type = "replace"
        m = MISMATCHED_CHARS_RE.search(error_text)
        found, expected = (m.group(1), m.group(2)) if m else ("?", "?")
        solution = (
            f"This should be a closing '{expected}' to match the bracket opened earlier, but '{found}' "
            f"appears instead. Either replace it with '{expected}', or - if '{found}' is actually correct "
            f"here - check whether an earlier bracket in this block was closed at the wrong spot, since "
            f"that would make this one line up with the wrong opener."
        )

    elif rule == "unexpected_closing_bracket":
        solution_type = "replace"
        m = UNEXPECTED_CHAR_RE.search(error_text)
        closer = m.group(1) if m else "}"
        solution = (
            f"This '{closer}' has no matching opening bracket anywhere before it in the file. Either "
            f"delete this extra '{closer}', or - if it's meant to close something real - add the missing "
            f"opening bracket earlier in the code where that block, call, or expression actually starts."
        )

    elif rule == "unterminated_string":
        solution_type = "replace"
        solution = "Add the missing closing quote at the end of this string - it needs to match whichever quote character (' or \") opened it. A string can't span multiple lines unless it's a template literal (backticks) or a triple-quoted string, so a missing quote here usually means the string was meant to end on this same line."

    elif rule == "unterminated_template_literal":
        solution_type = "replace"
        solution = "Add the missing closing backtick (`) to complete this template string. Every backtick that opens a template literal needs exactly one matching backtick to close it - count the backticks on this line and nearby lines to find where one was left out."

    elif rule == "unterminated_comment":
        solution_type = "replace"
        solution = "Add the missing */ to close this comment block. Until it's closed, every line after it in the file is silently treated as part of the comment, which can hide real code from the compiler without any warning - so check that nothing important got swallowed once this is fixed."

    elif rule == "loose_equality":
        solution_type = "replace"
        if "!=" in current_code:
            replacement_code = current_code.replace("!=", "!==")
            solution = "Change '!=' to '!==', as shown below. '!=' compares values after converting them to a common type first, which can make surprisingly different values look equal (e.g. 0 != \"0\" is false) - '!==' compares type and value together with no conversion."
        else:
            replacement_code = current_code.replace("==", "===")
            solution = "Change '==' to '===', as shown below. '==' compares values after converting them to a common type first, which can make surprisingly different values look equal (e.g. 0 == \"0\" is true) - '===' compares type and value together with no conversion."

    elif rule == "var_declaration":
        solution_type = "replace"
        replacement_code = current_code.replace("var ", "let ", 1)
        solution = "Change 'var' to 'let' (or 'const' if this value is never reassigned), as shown below. 'var' is function-scoped and hoisted, which can let a variable leak out of the block it looks like it belongs to - 'let'/'const' are block-scoped and avoid that entire class of bug."

    elif rule == "empty_catch_block":
        replacement_code = _fix_empty_catch(current_code, finding.get("file", ""))
        if replacement_code:
            solution_type = "replace"
            solution = "Replace this with the version below, which logs the caught error instead of silently discarding it. Right now, if this code ever throws, the failure disappears with no log, no fallback, and no way to know it happened."
        else:
            solution_type = "add"
            solution = "Add at least a log statement (e.g. console.error(err), or the equivalent for this language) inside the catch block. Right now this catch block does nothing, so if the wrapped code ever throws, the failure is silently discarded with no trace of it happening."

    elif rule == "leftover_console_statement":
        solution_type = "remove"
        solution = "Delete this console.log/debug statement before shipping. It's harmless in production but usually isn't meant to ship, and can leak internal data into the browser console - if it's intentional logging rather than a debugging leftover, it's fine to leave as-is."

    elif rule == "leftover_debugger_statement":
        solution_type = "remove"
        solution = "Delete this 'debugger' statement before shipping. It pauses execution in any browser with developer tools open, which is almost always leftover from debugging rather than something meant to run in production."

    elif rule == "leftover_debug_print":
        solution_type = "remove"
        solution = "Delete this print statement before shipping, unless it's intentional program output (e.g. a CLI tool's actual result) rather than a debugging leftover - if you're not sure which it is, check whether removing it would change what the program is supposed to display to a real user."

    elif rule == "unreachable_code":
        solution_type = "remove"
        solution = "Delete this code. It sits right after a return, throw, or a branch that always exits, so it can never actually execute - removing it has no effect on the program's behavior, since it never ran in the first place."

    else:
        # Should not normally be reached - every known rule is handled above -
        # but keep a safe, honest fallback for any future/unknown rule.
        solution_type = "replace"
        solution = "We couldn't prepare an automatic fix for this one - review the code below and apply the fix yourself, using the cause above as a starting point."

    return {
        "error": finding.get("error", "Possible issue"),
        "bug_type": bug_type,
        "cause": cause or "Something in this code looks like it could cause a problem.",
        "why_occurs": "",
        "solution_type": solution_type,
        "solution": solution,
        "replacement_code": replacement_code,
        "add_location": add_location,
        "new_file_path": None,
        "explanation": "",
        "confidence": FALLBACK_CONFIDENCE_BY_RULE.get(rule, DEFAULT_FALLBACK_CONFIDENCE),
        "insufficient_evidence": True,
    }


def get_fallback_report(finding: dict) -> dict:
    """
    Public entry point to the same real, rule-specific advice used when
    the LLM is unavailable - lets a caller intentionally skip the API call
    (e.g. once a per-scan call budget is used up) while still returning a
    genuine, specific answer instead of dropping the finding entirely.
    """
    return _fallback_report(finding)


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

                    # A missing/empty solution would mean the one thing the person
                    # actually opens this report to read is blank - never let that
                    # reach the frontend silently.
                    if not (result.get("solution") or "").strip():
                        print(f"[llm_client] Gemini returned an empty solution field. Using fallback.")
                        return _fallback_report(finding)

                    # Gemini occasionally omits confidence despite the instruction, or
                    # returns something outside 0-100 - never let a missing/bad number
                    # here break the scan-wide average computed in main.py.
                    confidence = result.get("confidence")
                    if not isinstance(confidence, (int, float)) or not (0 <= confidence <= 100):
                        result["confidence"] = 70
                    else:
                        result["confidence"] = int(confidence)

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

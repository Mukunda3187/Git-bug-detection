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

# See the comment in analyze_finding() where this is used - together these two
# bound the worst-case time one finding's LLM call can take to
# MAX_KEYS_TO_TRY_PER_CALL * GEMINI_REQUEST_TIMEOUT_SECONDS, regardless of how
# many keys end up configured.
MAX_KEYS_TO_TRY_PER_CALL = 4
GEMINI_REQUEST_TIMEOUT_SECONDS = 7


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

SYSTEM_PROMPT = """You are an expert software debugger. You are given ONE individual bug finding at a time.

Your job is NOT to give a generic explanation for the bug category. You must analyze THIS exact finding independently.

Use ALL of these inputs:
- exact file path
- exact function/component
- exact line range
- detector rule
- detector's cause
- exact Current Code snippet
- relevant historical bug/fix examples from the RAG context

FIRST analyze the exact Current Code.
Then identify the specific mistake in that code.
Then create the smallest reliable fix for THAT exact code.
Do not copy a solution from another bug just because the rule name is similar.

OUTPUT REQUIREMENTS:

1. "cause"
Explain why THIS exact code is wrong in 1-2 short sentences. Mention the actual variable, statement, operator, bracket, function, or code pattern involved.

2. "solution_type"
Choose exactly one:
- "replace" = the shown buggy code should be replaced by corrected code
- "add" = new code must be added
- "remove" = the shown code should be deleted
- "create_file" = a new file must be created

3. "replacement_code"
For "replace":
- Return the complete corrected replacement for Current Code.
- It MUST be different from Current Code.
- Keep unrelated code unchanged.
- The code must directly fix THIS finding.
- Do not return a generic example.

For "add":
- Return the exact code that should be added, based on THIS finding.

For "remove":
- Set replacement_code to null. Never repeat the buggy Current Code as the solution.

For "create_file":
- Return the exact new file content when enough context exists.

4. "solution"
Write EXACTLY TWO short sentences and make them specific to THIS bug.
Sentence 1 must tell the developer exactly what to change.
Sentence 2 must tell briefly why that exact change fixes THIS bug.
Do NOT write generic theory.
Do NOT reuse the same sentence for different bugs.
Do NOT mention generic advice such as "review the code", "follow best practices", or "handle errors properly".
Do NOT put code blocks in solution.

5. "explanation"
One short technical sentence explaining why the generated fix works.

6. "confidence"
Integer 0-100 based on the actual evidence in THIS finding.

7. "insufficient_evidence"
Set true if the exact supplied code is not enough to safely generate the required fix. Never invent a fix.

IMPORTANT:
- Every finding is independent. Never assume two findings have the same solution.
- The same rule can have different fixes depending on the actual Current Code.
- The solution must be derived from the supplied code, not from the rule name alone.
- Historical RAG examples are supporting evidence only; adapt them to the Current Code.
- Never return replacement_code identical to Current Code.
- Never claim a fix is exact when the required context is missing.
- Return ONLY valid JSON. No markdown fences and no text outside JSON.

JSON schema:
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
Treat this as a NEW and INDEPENDENT bug. Analyze only this finding first, using the exact Current Code and context above.

Do not reuse a generic solution from another bug. Derive the fix from the actual statement, variable, operator, bracket, control flow, or other code shown here.

For "replace", replacement_code must be a genuinely corrected version and MUST differ from Current Code.
For "remove", replacement_code must be null.
The two solution sentences must describe what to do for THIS exact bug and why THIS exact change fixes it."""


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
        fn_match = re.search(
            r"\b(?:def|async\s+def|function)\s+([A-Za-z_]\w*)",
            current_code,
        )
        fn_name = fn_match.group(1) if fn_match else "this function"
        solution = (
            f"Remove `{fn_name}()` because this function is not called anywhere in the scanned file. "
            f"This removes the specific unused definition reported at this location without changing the calls detected in this file."
        )

    elif rule == "bare_except":
        solution_type = "replace"
        replacement_code = current_code.replace("except:", "except Exception as e:", 1)
        solution = (
            f"Replace the bare `except:` in {_code_anchor(finding)} with `except Exception as e:`. "
            f"This keeps ordinary exception handling for the reported block without swallowing system-exit signals."
        )

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
        solution = (
            f"Change the None comparison in {_code_anchor(finding)} from `==`/`!=` to `is`/`is not`. "
            f"This makes the reported None check use Python identity comparison as intended."
        )

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
            solution = (
                f"Replace `!=` in {_code_anchor(finding)} with `!==`. "
                f"This makes the reported comparison require both the same type and the same value."
            )
        else:
            replacement_code = current_code.replace("==", "===")
            solution = (
                f"Replace `==` in {_code_anchor(finding)} with `===`. "
                f"This makes the reported comparison require both the same type and the same value."
            )

    elif rule == "var_declaration":
        solution_type = "replace"
        replacement_code = current_code.replace("var ", "let ", 1)
        solution = (
            f"Replace the `var` declaration in {_code_anchor(finding)} with the corrected `let` declaration shown below. "
            f"This keeps the reported variable scoped to the block where it is used."
        )

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

    if solution_type == "replace" and isinstance(replacement_code, str):
        if re.sub(r"\s+", "", replacement_code) == re.sub(r"\s+", "", current_code):
            replacement_code = None
            solution_type = "add"
            add_location = add_location or "Edit the reported code at this location using the detected cause."
            solution = "Apply the correction described by the detected cause at this location. An exact automatic replacement could not be generated safely from the available code."

    fallback_confidence, fallback_level, fallback_status = _calculate_evidence_confidence(
        finding=finding,
        retrieved=[],
        llm_confidence=FALLBACK_CONFIDENCE_BY_RULE.get(
            rule, DEFAULT_FALLBACK_CONFIDENCE
        ),
        insufficient_evidence=True,
    )

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
        "confidence": fallback_confidence,
        "confidence_level": fallback_level,
        "confidence_status": fallback_status,
        "insufficient_evidence": True,
    }



def analyze_file(file_path: str, source: str) -> list:
    """
    Analyze a readable text file directly with Gemini when there is no
    specialized detector for its extension.

    This is intentionally separate from analyze_finding(): analyze_finding()
    receives one detector finding plus RAG context, while this function asks
    Gemini to inspect the whole file and return candidate findings that can
    then enter the normal RAG + report pipeline in main.py.

    Returns an empty list when no API key is available, the file is empty, the
    model call fails, or Gemini returns invalid JSON. This keeps the normal
    detector/fallback pipeline safe instead of inventing a finding.
    """
    if not source or not source.strip():
        return []

    api_keys = []

    # Reuse the same multi-key configuration as analyze_finding().
    for i in range(1, 11):
        key = os.getenv(f"GEMINI_API_KEY_{i}")
        if key and key.strip():
            api_keys.append(key.strip())

    old_key = os.getenv("GEMINI_API_KEY")
    if old_key and old_key.strip() and old_key.strip() not in api_keys:
        api_keys.append(old_key.strip())

    if not api_keys:
        return []

    api_keys = api_keys[:MAX_KEYS_TO_TRY_PER_CALL]

    file_prompt = f"""You are reviewing a source/configuration/text file for real software bugs.

File: {file_path}

Analyze the complete file below. Report ONLY issues that are reasonably supported
by the code/text itself. Do not invent bugs just because a style preference is
not followed. If there are no clear issues, return an empty JSON array.

For every real or strongly supported issue, return an object with exactly these
fields:
- error: short description of the bug
- bug_type: one of Runtime Error, Logic Error, Syntax Error, Type Error,
  Dependency Error, Security Issue, Performance Issue, API Error,
  Unnecessary Code, Other
- cause: why the issue occurs
- line_start: 1-based starting line number
- line_end: 1-based ending line number
- current_code: the smallest relevant code/text snippet
- function: function/class/component name if applicable, otherwise null

Rules:
- Use 1-based line numbers.
- Only report issues you can point to in the supplied file.
- Do not report vague possibilities without evidence.
- Do not include markdown fences or explanations outside the JSON array.

Return ONLY a JSON array.

FILE CONTENT:
--------------------
{source}
--------------------
"""

    payload = {
        "system_instruction": {
            "parts": [{"text": "You are a precise software bug detector. Return valid JSON only."}]
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": file_prompt}],
            }
        ],
        "generationConfig": {
            "response_mime_type": "application/json",
            "temperature": 0.1,
            "maxOutputTokens": 2500,
            "topP": 0.9,
            "topK": 40,
        },
    }

    last_error = None

    for key_number, api_key in enumerate(api_keys, start=1):
        try:
            resp = requests.post(
                GEMINI_URL,
                params={"key": api_key},
                json=payload,
                timeout=GEMINI_REQUEST_TIMEOUT_SECONDS,
            )

            if resp.status_code != 200:
                last_error = (
                    f"Gemini file-analysis key {key_number} returned "
                    f"HTTP {resp.status_code}: {resp.text[:200]}"
                )
                continue

            data = resp.json()
            raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
            cleaned = (
                raw_text.strip()
                .removeprefix("```json")
                .removeprefix("```")
                .removesuffix("```")
                .strip()
            )

            result = json.loads(cleaned)
            if not isinstance(result, list):
                print("[llm_client] Gemini file analysis did not return a JSON array.")
                return []

            findings = []
            for item in result:
                if not isinstance(item, dict):
                    continue

                error = str(item.get("error") or "").strip()
                if not error:
                    continue

                bug_type = str(item.get("bug_type") or "Other").strip()
                allowed_types = {
                    "Runtime Error", "Logic Error", "Syntax Error", "Type Error",
                    "Dependency Error", "Security Issue", "Performance Issue",
                    "API Error", "Unnecessary Code", "Other",
                }
                if bug_type not in allowed_types:
                    bug_type = "Other"

                line_start = item.get("line_start")
                line_end = item.get("line_end")
                try:
                    line_start = int(line_start) if line_start is not None else None
                    line_end = int(line_end) if line_end is not None else line_start
                except (TypeError, ValueError):
                    line_start = None
                    line_end = None

                findings.append({
                    "error": error,
                    "bug_type": bug_type,
                    "cause": str(item.get("cause") or "").strip(),
                    "line_start": line_start,
                    "line_end": line_end,
                    "current_code": str(item.get("current_code") or "").strip(),
                    "function": item.get("function"),
                    "rule": "llm_file_analysis",
                    "file": file_path,
                })

            return findings

        except Exception as e:
            last_error = f"Gemini file analysis key {key_number} failed: {e}"

    if last_error:
        print(f"[llm_client] File analysis failed: {last_error}")
    return []

def get_fallback_report(finding: dict) -> dict:
    """
    Public entry point to the same real, rule-specific advice used when
    the LLM is unavailable - lets a caller intentionally skip the API call
    (e.g. once a per-scan call budget is used up) while still returning a
    genuine, specific answer instead of dropping the finding entirely.
    """
    return _fallback_report(finding)



def _code_anchor(finding: dict) -> str:
    """Return a small, human-readable identifier from THIS finding's code."""
    code = str(finding.get("current_code") or "").strip()
    if not code:
        return "the reported code"

    # Prefer a function/class name because it makes the solution visibly
    # specific to the individual finding.
    m = re.search(r"\b(?:def|async\s+def|function|class)\s+([A-Za-z_]\w*)", code)
    if m:
        return f"`{m.group(1)}()`"

    # Otherwise use the first meaningful code line, shortened for the UI.
    first = next((x.strip() for x in code.splitlines() if x.strip()), code)
    first = re.sub(r"\\s+", " ", first)
    if len(first) > 70:
        first = first[:67] + "..."
    return f"`{first}`"


def _solution_is_specific(solution: str, finding: dict) -> bool:
    """Require the solution to refer to something actually present in this finding."""
    if not solution:
        return False

    code = str(finding.get("current_code") or "")
    if not code:
        return True

    # Exact function/class name is the strongest signal.
    names = re.findall(r"\b(?:def|async\s+def|function|class)\s+([A-Za-z_]\w*)", code)
    for name in names:
        if re.search(rf"\b{re.escape(name)}\b", solution):
            return True

    # Require at least one meaningful identifier/operator/token from the
    # actual code, rather than allowing "this code" / "replace this" templates.
    tokens = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b", code)
    stop = {
        "def", "return", "function", "class", "self", "true", "false",
        "none", "null", "this", "else", "elif", "from", "import", "with",
        "async", "await", "try", "except", "finally", "for", "while", "if",
    }
    meaningful = [t for t in tokens if t.lower() not in stop]
    return any(re.search(rf"\b{re.escape(t)}\b", solution) for t in meaningful[:20])


def _make_specific_solution(result: dict, finding: dict) -> str:
    """Create a short solution tied to this exact finding when Gemini is generic."""
    anchor = _code_anchor(finding)
    solution_type = result.get("solution_type")

    if solution_type == "remove":
        return (
            f"Remove {anchor} from this file because this reported code is not used here. "
            f"This removes the specific unused code without changing the code path shown in the finding."
        )

    if solution_type == "add":
        return (
            f"Add the required check or statement at the reported location for {anchor}. "
            f"This prevents the specific condition identified in this finding from reaching the failing code."
        )

    if solution_type == "create_file":
        return (
            f"Create the new file shown in the generated solution for {anchor}. "
            f"This supplies the missing file or definition required by the reported code."
        )

    return (
        f"Replace {anchor} with the corrected code shown in the Solution section. "
        f"This changes the exact code reported by the detector while preserving the surrounding code."
    )


def _short_solution(text: str) -> str:
    """Return only the first two useful sentences for the Solution UI."""
    if not text:
        return ""
    clean = re.sub(r"```(?:\w+)?|```", "", str(text)).strip()
    clean = re.sub(r"\s+", " ", clean)
    sentences = re.split(r"(?<=[.!?])\s+", clean)
    sentences = [s.strip() for s in sentences if s.strip()]
    return " ".join(sentences[:2]) if sentences else clean


def _same_code(a, b) -> bool:
    """Compare code while ignoring harmless whitespace differences."""
    if not isinstance(a, str) or not isinstance(b, str):
        return False
    normalize = lambda s: re.sub(r"\s+", "", s).strip()
    return normalize(a) == normalize(b)



def _clamp_confidence(value: float) -> int:
    """Keep confidence in the valid 0-100 range."""
    return max(0, min(100, int(round(value))))


def _detector_evidence_score(finding: dict) -> int:
    """
    Estimate how strong the detector evidence is for THIS finding.

    This is evidence strength, not a claim that the detector is statistically
    calibrated. Exact syntax/structural findings receive stronger evidence than
    heuristic findings.
    """
    rule = str(finding.get("rule") or "").lower()

    strong_rules = {
        "syntax_error",
        "unclosed_bracket",
        "mismatched_bracket",
        "unexpected_closing_bracket",
        "unterminated_string",
        "unterminated_template_literal",
        "unterminated_comment",
        "unreachable_code",
        "eq_none",
    }

    medium_rules = {
        "bare_except",
        "mutable_default_arg",
        "loose_equality",
        "var_declaration",
        "leftover_console_statement",
        "leftover_debugger_statement",
        "leftover_debug_print",
        "possibly_unused_function",
        "possible_division_by_zero",
        "empty_catch_block",
    }

    if rule in strong_rules:
        return 95
    if rule in medium_rules:
        return 80

    # LLM-discovered findings have weaker detector evidence because they were
    # not produced by a specialized deterministic rule.
    if rule == "llm_file_analysis":
        return 65

    return 60


def _rag_evidence_score(retrieved: list) -> int:
    """
    Convert the strongest retrieved historical similarity into an evidence
    score. No retrieved evidence contributes zero.
    """
    if not retrieved:
        return 0

    similarities = []
    for item in retrieved:
        try:
            value = float(item.get("similarity", 0.0))
        except (TypeError, ValueError):
            value = 0.0
        similarities.append(max(0.0, min(1.0, value)))

    if not similarities:
        return 0

    return _clamp_confidence(max(similarities) * 100)


def _calculate_evidence_confidence(
    finding: dict,
    retrieved: list,
    llm_confidence,
    insufficient_evidence: bool = False,
) -> tuple[int, str, str]:
    """
    Calculate the final per-bug confidence from three explicit evidence
    sources:

      50% = LLM confidence for this exact finding
      30% = static detector evidence strength
      20% = strongest historical RAG similarity

    This is a transparent evidence score for the project UI. It is not a
    statistically calibrated probability unless the project later validates
    it against labelled bug data.
    """
    try:
        llm_score = float(llm_confidence)
    except (TypeError, ValueError):
        llm_score = 50.0

    llm_score = max(0.0, min(100.0, llm_score))
    detector_score = _detector_evidence_score(finding)
    rag_score = _rag_evidence_score(retrieved)

    final_score = (
        llm_score * 0.50
        + detector_score * 0.30
        + rag_score * 0.20
    )

    if insufficient_evidence:
        final_score = min(final_score, 49)

    confidence = _clamp_confidence(final_score)

    if confidence >= 70 and not insufficient_evidence:
        level = "High Confidence"
        status = "Potential Bug"
    else:
        level = "Low Confidence"
        status = "Uncertain Finding"

    return confidence, level, status


def _validate_llm_result(result: dict, finding: dict, retrieved: list | None = None):
    """Reject unusable or non-specific LLM fixes before they reach the frontend."""
    if not isinstance(result, dict):
        return None

    solution_type = result.get("solution_type")
    current_code = str(finding.get("current_code") or "")
    replacement = result.get("replacement_code")

    if solution_type == "replace":
        if not isinstance(replacement, str) or not replacement.strip():
            return None
        if _same_code(replacement, current_code):
            print("[llm_client] Gemini returned the same code as the error code. Using fallback.")
            return None

    if solution_type == "remove":
        # A remove solution must never display the same buggy code again.
        result["replacement_code"] = None

    result["solution"] = _short_solution(result.get("solution", ""))

    if not result["solution"]:
        return None

    # A solution such as "Delete this function" can be technically valid but
    # is too generic for the UI. It must refer to the actual function,
    # variable, statement, or other token in THIS finding.
    if not _solution_is_specific(result["solution"], finding):
        result["solution"] = _make_specific_solution(result, finding)

    result["solution"] = _short_solution(result["solution"])

    result.setdefault("insufficient_evidence", False)

    confidence, confidence_level, confidence_status = _calculate_evidence_confidence(
        finding=finding,
        retrieved=retrieved or [],
        llm_confidence=result.get("confidence"),
        insufficient_evidence=bool(result.get("insufficient_evidence", False)),
    )
    result["confidence"] = confidence
    result["confidence_level"] = confidence_level
    result["confidence_status"] = confidence_status

    return result


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

    # Bounds the worst case for one finding to MAX_KEYS_TO_TRY_PER_CALL * the
    # per-attempt timeout below, no matter how many keys end up configured -
    # a scan with many findings calls this once per finding, so an unbounded
    # key list here directly multiplies into minutes of wall-clock time on a
    # bad run (several keys rate-limited or slow to respond) even before
    # accounting for how many findings there are.
    api_keys = api_keys[:MAX_KEYS_TO_TRY_PER_CALL]

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
                timeout=GEMINI_REQUEST_TIMEOUT_SECONDS,
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

                    validated = _validate_llm_result(result, finding, retrieved)
                    if validated is None:
                        return _fallback_report(finding)
                    return validated

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

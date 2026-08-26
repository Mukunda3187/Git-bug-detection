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
import os

import requests

GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

SYSTEM_PROMPT = """You are a careful code-review assistant helping a bug-detection tool.
You will be given ONE candidate issue found by a static analyzer in a real source file,
plus a small number of similar historical bugs retrieved from a knowledge base.

Your job:
1. Decide whether this candidate is worth reporting as a possible bug, given the evidence.
2. If yes, explain the cause AND why it likely happened (e.g. leftover from refactoring,
   missing validation, unsafe string handling), based on the ACTUAL code shown - never
   invent unrelated code.
3. If solution_type is "replace", you MUST always provide a non-null, non-empty
   replacement_code with the actual corrected code - never leave it null for a
   replace solution. If you cannot write a concrete fix, use solution_type "add"
   or set insufficient_evidence to true instead of leaving replacement_code empty.
   CRITICAL: replacement_code must be a direct, drop-in replacement for exactly
   the current_code shown - the developer must be able to find current_code in
   their file and swap it for replacement_code verbatim. Keep the same variable
   names, same structure, same surrounding logic - change ONLY what is actually
   broken. Do not paraphrase unrelated parts, rename variables, or restructure
   code that isn't part of the bug.
4. Pick exactly ONE solution_type that matches what the developer needs to do:
   - "replace": existing code is wrong and should be swapped for corrected code
   - "add": nothing is wrong with existing code, but a check/line is missing and needs adding
   - "remove": the code should simply be deleted (e.g. unused/dead code)
   - "create_file": a new file is needed (rare - only use this if that's genuinely the fix)
5. Write solution_intro as the exact required first sentence for that solution_type:
   - replace -> "Replace the given code with the new code shown below."
   - add -> "Add the following code as instructed below."
   - remove -> "Remove the given code."
   - create_file -> "Create the file at the location given below and paste the following code into it."
6. Write action as ONE short final sentence telling the developer exactly what to do, e.g.
   "Replace the current code with the code shown above." or "Remove this code from the file."
7. If the evidence is weak (e.g. only a vague heuristic match, no real historical support), set
   "insufficient_evidence" to true and explain why in "explanation", rather than guessing.

Reply with ONLY a JSON object, no markdown fences, no extra text, with exactly these keys:
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
}
"""


def _build_user_message(finding: dict, retrieved: list) -> str:
    retrieved_block = "\n\n".join(
        f"- Dataset: {r['record'].get('dataset_source')}\n"
        f"  Bug type: {r['record'].get('bug_type')}\n"
        f"  Description: {r['record'].get('bug_description')}\n"
        f"  Similarity: {round(r['similarity'] * 100)}%"
        for r in retrieved
    ) or "(no similar historical bugs found)"

    return f"""Candidate issue detected by static analyzer:
File: {finding.get('file')}
Function: {finding.get('function')}
Line(s): {finding.get('line_start')}-{finding.get('line_end')}
Detector rule: {finding.get('rule')}
Detector's initial label: {finding.get('error')} ({finding.get('bug_type')})
Detector's initial cause note: {finding.get('cause')}

Code:
```
{finding.get('current_code')}
```

Similar historical bugs retrieved from the knowledge base:
{retrieved_block}

Now produce the JSON bug report."""


def _fallback_report(finding: dict) -> dict:
    """Used when no LLM reasoning is available - either no API key is configured, or the
    live LLM call failed/timed out (analyze_finding overwrites `explanation` for that case).
    Never claims to show a "new code below" it doesn't actually have."""
    is_unused = finding.get("rule") == "possibly_unused_function"
    return {
        "error": finding.get("error", "Possible issue"),
        "bug_type": finding.get("bug_type", "Other"),
        "cause": finding.get("cause", ""),
        "why_occurs": "This was flagged by static analysis rules; no LLM reasoning is available right now.",
        "solution_type": "remove" if is_unused else "replace",
        "solution_intro": (
            "Remove the given code." if is_unused else
            "No automated fix could be generated right now (see note below) - review the code "
            "shown below and apply a fix manually."
        ),
        "replacement_code": None,
        "add_location": None,
        "new_file_path": None,
        "action": "Remove this code from the file." if is_unused else "Review this code and apply a fix manually.",
        "explanation": "No LLM API key configured (GEMINI_API_KEY missing) - showing the raw static "
                       "analyzer finding without LLM reasoning or a suggested fix.",
        "insufficient_evidence": True,
    }


def analyze_finding(finding: dict, retrieved: list) -> dict:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return _fallback_report(finding)

    payload = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": _build_user_message(finding, retrieved)}]}],
        "generationConfig": {
            "response_mime_type": "application/json",
            "temperature": 0.2,
            "maxOutputTokens": 1000,
        },
    }

    try:
        resp = requests.post(
            GEMINI_URL,
            params={"key": api_key},
            json=payload,
            timeout=30,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Gemini API returned status {resp.status_code}: {resp.text[:300]}")

        data = resp.json()
        raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        # LLM call failed for any reason (bad key, rate limit, network) -
        # never let this crash the whole scan.
        fallback = _fallback_report(finding)
        fallback["explanation"] = f"LLM call failed ({e}). Showing the raw static analyzer result instead."
        return fallback

    try:
        cleaned = raw_text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        fallback = _fallback_report(finding)
        fallback["explanation"] = "The LLM response could not be parsed as JSON, so this finding is shown " \
                                   "using the raw static analyzer result instead."
        return fallback

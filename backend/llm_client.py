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

SYSTEM_PROMPT = """You are helping explain code bugs to someone who may have very little
programming knowledge - a beginner, or someone who has never coded at all. You will be
given ONE candidate issue found by a static analyzer in a real source file, plus a small
number of similar historical bugs retrieved from a knowledge base (for your own context only
- do not mention this knowledge base to the reader).

Your job:
1. Decide whether this candidate is worth reporting as a possible bug, given the evidence.
2. Write "cause" in EXTREMELY SIMPLE English - no jargon, no technical terms a non-programmer
   wouldn't know. It must explain BOTH what is wrong AND why that causes a problem, in plain
   everyday words. Example of the required style:
   Bad (too technical): "An unmatched delimiter causes a parser failure."
   Good (required style): "You opened a { bracket, but you did not close it with }. Because
   of this, the computer cannot understand where this part of the code ends."
   Always write "cause" in this same plain, friendly, step-by-step style - imagine explaining
   it to a smart 12-year-old who has never programmed before.
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
5. Write solution_intro in simple, plain English that tells the reader exactly what to do,
   using the required first sentence for that solution_type:
   - replace -> "Replace the code shown above with the corrected code below."
   - add -> "Add the following code as shown below."
   - remove -> "Remove this code from the file."
   - create_file -> "Create this file at the location shown below and paste the following code into it."
   If solution_type is "add", also fill add_location with a simple plain-English description of
   exactly where to add the code, e.g. "Add this code between lines 20 and 25." or "Add this
   code right after the function called validate_user finishes."
6. Write "action" as ONE short, extremely simple final sentence telling the reader exactly what
   to do, in plain English, e.g. "Add the missing closing bracket shown above." or "Delete this
   code from the file." Never say something vague like "fix the issue" or "modify the function
   accordingly" - always say exactly what to add, remove, or replace.
7. If the evidence is weak (e.g. only a vague heuristic match, no real historical support), set
   "insufficient_evidence" to true, and still explain in "cause" what looks suspicious, in the
   same simple plain English style.

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
    live LLM call failed/timed out. Kept in plain, non-technical English since this is
    shown directly to the end user - technical failure details are logged to the server
    console instead (see analyze_finding), never shown in the UI."""
    is_unused = finding.get("rule") == "possibly_unused_function"
    return {
        "error": finding.get("error", "Possible issue"),
        "bug_type": finding.get("bug_type", "Other"),
        "cause": finding.get("cause", "Something in this code looks like it could cause a problem."),
        "why_occurs": "",
        "solution_type": "remove" if is_unused else "replace",
        "solution_intro": (
            "Remove this code from the file." if is_unused else
            "We could not prepare an automatic fix for this one right now - please look at "
            "the code below and fix it yourself."
        ),
        "replacement_code": None,
        "add_location": None,
        "new_file_path": None,
        "action": "Remove this code from the file." if is_unused else "Review this code and fix it yourself.",
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
            "temperature": 0.2,
            "maxOutputTokens": 1000,
        },
    }

    last_error = None

    # Try each Gemini API key until one succeeds.
    for key_number, api_key in enumerate(api_keys, start=1):
        try:
            resp = requests.post(
                GEMINI_URL,
                params={"key": api_key},
                json=payload,
                timeout=30,
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

                    return json.loads(cleaned)

                except (json.JSONDecodeError, ValueError):
                    print(f"[llm_client] Gemini returned non-JSON response: {raw_text[:300]}")
                    return _fallback_report(finding)

            # Key failed - try the next key. Log the technical detail server-side
            # only - the person using the app should never see raw HTTP/API errors.
            last_error = f"Gemini key {key_number} returned HTTP {resp.status_code}: {resp.text[:200]}"

        except Exception as e:
            last_error = f"Gemini key {key_number} failed: {e}"

    # All keys failed. Print the real reason to the server logs (visible in Render's
    # Logs tab) so it can still be debugged, but keep the user-facing result simple.
    print(f"[llm_client] All Gemini API keys failed. Last error: {last_error}")
    return _fallback_report(finding)

"""
Wraps the call to the LLM (Claude by default). Give it:
  - a raw candidate finding from a detector (detectors/python_detector.py)
  - the retrieved similar historical bugs from the RAG step
and it returns a fully-formed structured bug report as a dict, matching
models.BugReport (minus the fields the caller already knows, like id/file).

If no ANTHROPIC_API_KEY is set, falls back to a transparent rule-based
formatter so the app still runs end-to-end for a demo - it clearly does
NOT pretend to be the LLM's reasoning, it just formats what the detector
already found.
"""
import json
import os

SYSTEM_PROMPT = """You are a careful code-review assistant helping a bug-detection tool.
You will be given ONE candidate issue found by a static analyzer in a real source file,
plus a small number of similar historical bugs retrieved from a knowledge base.

Your job:
1. Decide whether this candidate is worth reporting as a possible bug, given the evidence.
2. If yes, classify it and suggest a concrete fix based on the ACTUAL code shown - never invent unrelated code.
3. If the evidence is weak (e.g. only a vague heuristic match, no real historical support), set
   "insufficient_evidence" to true and explain why in "explanation", rather than guessing.

Reply with ONLY a JSON object, no markdown fences, no extra text, with exactly these keys:
{
  "error": string,
  "bug_type": one of ["Runtime Error","Logic Error","Syntax Error","Type Error","Dependency Error","Security Issue","Performance Issue","API Error","Unnecessary Code","Other"],
  "status_category": one of ["Easy Error","Frequent Error","Complex Error","API Error","Runtime Error","Logic Error","Syntax Error","Type Error","Dependency Error","Security Issue","Performance Issue","Other"],
  "severity": one of ["Low","Medium","High","Critical"],
  "confidence": integer 0-100,
  "cause": string,
  "replacement_code": string or null,
  "explanation": string,
  "action": string or null,
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
    """Used only when no LLM API key is configured - clearly a formatter, not an analysis."""
    return {
        "error": finding.get("error", "Possible issue"),
        "bug_type": finding.get("bug_type", "Other"),
        "status_category": finding.get("bug_type", "Other"),
        "severity": "Medium",
        "confidence": 50,
        "cause": finding.get("cause", ""),
        "replacement_code": None,
        "explanation": "No LLM API key configured (ANTHROPIC_API_KEY missing) - showing the raw static "
                       "analyzer finding without LLM reasoning or a suggested fix.",
        "action": "Remove this code." if finding.get("rule") == "possibly_unused_function" else None,
        "insufficient_evidence": True,
    }


def analyze_finding(finding: dict, retrieved: list) -> dict:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return _fallback_report(finding)

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    try:
        message = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _build_user_message(finding, retrieved)}],
        )
        raw_text = "".join(block.text for block in message.content if hasattr(block, "text"))
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

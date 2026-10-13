"""
Direct integration test of RAG retrieval in the analysis flow.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from rag.retriever import retrieve_similar_bugs
from models import RetrievedBug

# Simulate a finding with syntax error
finding = {
    "file": "test.py",
    "function": "greet",
    "line_start": 3,
    "line_end": 3,
    "rule": "unclosed_bracket",
    "bug_type": "Syntax Error",
    "error": "SyntaxError: '(' was never closed",
    "cause": "Missing closing parenthesis in print statement",
    "current_code": 'def greet(name):\n    message = "Hello, " + name\n    print(message',
    "code_chunk": 'def greet(name):\n    message = "Hello, " + name\n    print(message',
    "chunk_context": "function greet\nLines 1-3\ndef greet(name):\n    message = \"Hello, \" + name\n    print(message"
}

print("=" * 70)
print("RAG INTEGRATION TEST")
print("=" * 70)

# Build query like main.py does
query_text = (
    f"{finding.get('error', '')}\n"
    f"{finding.get('cause', '')}\n"
    f"{finding.get('current_code', '')}\n"
    f"{finding.get('bug_type', '')}\n"
    f"{finding.get('chunk_context', '')}"
)

print("\n1. Query text:")
print("-" * 70)
print(query_text)
print()

# Retrieve similar bugs
print("2. Retrieving similar bugs...")
retrieved = retrieve_similar_bugs(query_text=query_text, top_k=3)

print(f"   Found {len(retrieved)} similar bugs")
print()

# Convert to RetrievedBug objects like main.py does
print("3. Converting to RetrievedBug objects...")
retrieved_bugs = []

for item in retrieved:
    record = item.get("record", {})
    source_type = item.get("source_type", "dataset")
    
    if source_type == "dataset":
        rb = RetrievedBug(
            dataset_source=record.get("dataset_source", "Historical Dataset"),
            bug_type=record.get("bug_type"),
            bug_description=record.get("bug_description"),
            solution=record.get("solution"),
            similarity=round(item.get("similarity", 0) * 100, 1),
            historical_file=record.get("file") or record.get("filename") or record.get("historical_file"),
            historical_code=record.get("code") or record.get("buggy_code") or record.get("historical_code"),
            historical_patch=record.get("patch") or record.get("diff"),
            artifact_type=record.get("artifact_type") or "dataset",
            evidence=record.get("evidence"),
        )
        retrieved_bugs.append(rb)
        print(f"   ✓ {rb.dataset_source}: {rb.similarity}% similar")
        print(f"     Bug Type: {rb.bug_type}")
        print(f"     Description: {rb.bug_description[:60]}...")
        print(f"     Solution: {rb.solution[:60]}..." if rb.solution else "     (no solution)")
        print()

print(f"4. Final count: {len(retrieved_bugs)} RetrievedBug objects")

if retrieved_bugs:
    print("\n✓ RAG INTEGRATION WORKING!")
    print("  The retrieved_bugs list has data and should appear in the frontend.")
else:
    print("\n✗ PROBLEM: No bugs were retrieved or converted!")

print("\n" + "=" * 70)

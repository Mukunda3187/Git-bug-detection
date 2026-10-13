"""
Test RAG retrieval with a real finding structure to debug matching.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from rag.retriever import retrieve_similar_bugs

# Simulate a real finding from the detector
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

# Current query construction from main.py
query_text = (
    f"{finding.get('error', '')}\n"
    f"{finding.get('cause', '')}\n"
    f"{finding.get('current_code', '')}\n"
    f"{finding.get('bug_type', '')}\n"
    f"{finding.get('chunk_context', '')}"
)

print("=" * 70)
print("CURRENT QUERY TEXT:")
print("=" * 70)
print(query_text)
print()

print("=" * 70)
print("RETRIEVAL RESULTS:")
print("=" * 70)

results = retrieve_similar_bugs(query_text, top_k=5)

if results:
    for i, result in enumerate(results, 1):
        record = result.get("record", {})
        similarity = result.get("similarity", 0)
        
        print(f"\n{i}. Similarity: {similarity * 100:.2f}%")
        print(f"   Source: {record.get('dataset_source', 'Unknown')}")
        print(f"   Error: {record.get('error', 'N/A')}")
        print(f"   Bug Type: {record.get('bug_type', 'Unknown')}")
        print(f"   Language: {record.get('language', 'N/A')}")
else:
    print("\nNO RESULTS - Similarity too low!")

print("\n" + "=" * 70)
print("TESTING IMPROVED QUERY (with language):")
print("=" * 70)

# Improved query with language detection
import os
file_ext = os.path.splitext(finding.get('file', ''))[1]
language_map = {
    '.py': 'python',
    '.js': 'javascript', 
    '.jsx': 'javascript',
    '.ts': 'typescript',
    '.tsx': 'typescript',
    '.java': 'java',
    '.c': 'c',
    '.cpp': 'cpp',
    '.cs': 'csharp',
    '.go': 'go',
    '.php': 'php'
}
language = language_map.get(file_ext, 'code')

improved_query = (
    f"Language: {language}\n"
    f"Bug Type: {finding.get('bug_type', '')}\n"
    f"Error: {finding.get('error', '')}\n"
    f"Cause: {finding.get('cause', '')}\n"
    f"Code:\n{finding.get('current_code', '')}"
)

print(improved_query)
print()

results2 = retrieve_similar_bugs(improved_query, top_k=5)

if results2:
    for i, result in enumerate(results2, 1):
        record = result.get("record", {})
        similarity = result.get("similarity", 0)
        
        print(f"\n{i}. Similarity: {similarity * 100:.2f}%")
        print(f"   Source: {record.get('dataset_source', 'Unknown')}")
        print(f"   Error: {record.get('error', 'N/A')}")
        print(f"   Solution: {record.get('solution', 'N/A')[:60]}...")
else:
    print("\nSTILL NO RESULTS!")

print("\n" + "=" * 70)

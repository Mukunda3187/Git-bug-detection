"""
Test script to debug RAG retrieval.
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from rag.retriever import (
    retrieve_similar_bugs,
    get_retriever_status,
    _load_faiss_index,
    _ensure_metadata
)

print("=" * 60)
print("RAG RETRIEVAL TEST")
print("=" * 60)

# Check retriever status
print("\n1. Checking retriever status...")
status = get_retriever_status()
print(f"   Index loaded: {status['loaded']}")
print(f"   Records: {status['records']}")
print(f"   FAISS vectors: {status['faiss_vectors']}")
print(f"   Embedding dimension: {status['embedding_dimension']}")
print(f"   Embedding model: {status['embedding_model']}")

# Ensure metadata is loaded
print("\n2. Ensuring metadata is loaded...")
_ensure_metadata()
print("   Metadata loaded successfully")

# Load FAISS index
print("\n3. Loading FAISS index...")
index = _load_faiss_index()
if index:
    print(f"   FAISS index loaded: {index.ntotal} vectors")
else:
    print("   ERROR: FAISS index could not be loaded")
    sys.exit(1)

# Test query simulating a syntax error
print("\n4. Testing retrieval with syntax error query...")
query_text = """
File: test.py
Bug type: Syntax Error
Detection rule: unclosed_bracket
Error: SyntaxError: '(' was never closed
Cause: Missing closing parenthesis
Code: def greet(name):
    message = "Hello, " + name
    print(message
"""

print(f"\nQuery:\n{query_text}\n")

results = retrieve_similar_bugs(query_text, top_k=5)

print(f"\n5. Retrieval Results: {len(results)} found")
print("=" * 60)

if results:
    for i, result in enumerate(results, 1):
        record = result.get("record", {})
        similarity = result.get("similarity", 0)
        
        print(f"\nResult {i}:")
        print(f"  Similarity: {similarity * 100:.2f}%")
        print(f"  Source: {record.get('dataset_source', 'Unknown')}")
        print(f"  Bug Type: {record.get('bug_type', 'Unknown')}")
        print(f"  Error: {record.get('error', 'N/A')}")
        print(f"  Description: {record.get('bug_description', 'N/A')[:100]}...")
        print(f"  Solution: {record.get('solution', 'N/A')[:100]}...")
else:
    print("\nNO RESULTS FOUND!")
    print("Possible reasons:")
    print("  - Similarity threshold too high (currently > 0.05)")
    print("  - Query doesn't match indexed content")
    print("  - Embedding model not working properly")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)

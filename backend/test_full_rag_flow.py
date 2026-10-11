"""
Test the complete RAG flow from scan request to response.
"""
import requests
import json
import time

API_BASE = "http://localhost:8000"

print("=" * 70)
print("FULL RAG FLOW TEST")
print("=" * 70)

# Test repository with known Python syntax errors
test_repo = "https://github.com/octocat/Hello-World"

print(f"\n1. Scanning repository: {test_repo}")
print(f"   RAG enabled: True")

scan_request = {
    "repo_url": test_repo,
    "rag_enabled": True
}

try:
    # Start scan
    response = requests.post(
        f"{API_BASE}/api/scan",
        json=scan_request,
        timeout=10
    )
    
    if response.status_code != 200:
        print(f"\nERROR: Scan request failed with status {response.status_code}")
        print(response.text[:500])
        exit(1)
    
    data = response.json()
    scan_id = data.get("scan_id")
    
    if not scan_id:
        print("\nERROR: No scan_id returned")
        print(json.dumps(data, indent=2))
        exit(1)
    
    print(f"\n2. Scan started with ID: {scan_id}")
    print("   Polling for completion...")
    
    # Poll for results
    max_attempts = 60  # 5 minutes max
    attempt = 0
    
    while attempt < max_attempts:
        time.sleep(5)
        attempt += 1
        
        poll_response = requests.get(
            f"{API_BASE}/api/scan/{scan_id}",
            timeout=10
        )
        
        if poll_response.status_code != 200:
            print(f"\n   Poll attempt {attempt}: HTTP {poll_response.status_code}")
            continue
        
        poll_data = poll_response.json()
        status = poll_data.get("status", "unknown")
        progress = poll_data.get("progress", 0)
        
        print(f"\r   Progress: {progress}% - Status: {status}", end="", flush=True)
        
        if status == "completed":
            print("\n\n3. Scan completed!")
            result = poll_data.get("result")
            
            if not result:
                print("\nERROR: No result in completed scan")
                exit(1)
            
            bugs = result.get("bugs", [])
            print(f"   Found {len(bugs)} bugs")
            
            if bugs:
                print("\n4. Checking RAG data in bugs:")
                for i, bug in enumerate(bugs[:3], 1):  # Check first 3 bugs
                    retrieved = bug.get("retrieved_bugs", [])
                    print(f"\n   Bug #{i}: {bug.get('file', 'unknown')}")
                    print(f"   Error: {bug.get('error', 'N/A')[:60]}...")
                    print(f"   Retrieved bugs: {len(retrieved)}")
                    
                    if retrieved:
                        print("   ✓ RAG DATA FOUND!")
                        for j, rb in enumerate(retrieved[:2], 1):
                            print(f"      - {rb.get('dataset_source', 'Unknown')}: {rb.get('similarity', 0)}% similar")
                            print(f"        {rb.get('bug_type', 'N/A')}")
                    else:
                        print("   ✗ NO RAG DATA (retrieved_bugs is empty)")
            else:
                print("\n   No bugs found in this repository (might be clean)")
            
            break
        
        elif status == "failed":
            print(f"\n\nERROR: Scan failed")
            print(f"Error: {poll_data.get('error', 'Unknown error')}")
            exit(1)
    
    if attempt >= max_attempts:
        print("\n\nTIMEOUT: Scan did not complete in time")
        exit(1)

except requests.exceptions.RequestException as e:
    print(f"\n\nNETWORK ERROR: {e}")
    print("\nMake sure the backend is running on http://localhost:8000")
    exit(1)

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)

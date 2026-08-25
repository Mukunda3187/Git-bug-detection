"""
BugsInPy is distributed as a GitHub repo (not a Hugging Face dataset),
where each bug is a folder under:
    projects/<project_name>/bugs/<bug_id>/
containing:
    bug_patch.txt   -> unified diff of the fix (buggy -> fixed)
    bug_info.txt    -> key: value metadata about the bug

Run on a machine with internet access:
    git clone https://github.com/soarsmu/BugsInPy.git
    python normalize_bugsinpy.py --bugsinpy-path ./BugsInPy

Output: datasets/normalized/bugsinpy.jsonl
"""
import argparse
import json
import os
import re
import sys

sys.path.append(os.path.dirname(__file__))
from schema import NormalizedBug

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "normalized", "bugsinpy.jsonl")


def parse_bug_info(path: str) -> dict:
    info = {}
    if not os.path.exists(path):
        return info
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "=" in line:
                key, _, value = line.partition("=")
                info[key.strip()] = value.strip().strip('"')
    return info


def split_patch_into_before_after(patch_text: str):
    """
    Very small unified-diff splitter: pulls out '-' lines as buggy code
    and '+' lines as fixed code. Good enough for RAG context, not meant
    to be a full diff engine.
    """
    buggy_lines, fixed_lines = [], []
    for line in patch_text.splitlines():
        if line.startswith("-") and not line.startswith("---"):
            buggy_lines.append(line[1:])
        elif line.startswith("+") and not line.startswith("+++"):
            fixed_lines.append(line[1:])
    return "\n".join(buggy_lines).strip(), "\n".join(fixed_lines).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bugsinpy-path", required=True, help="Path to cloned BugsInPy repo")
    args = parser.parse_args()

    projects_dir = os.path.join(args.bugsinpy_path, "projects")
    if not os.path.isdir(projects_dir):
        print(f"Could not find 'projects' folder inside {args.bugsinpy_path}. "
              f"Did you clone the BugsInPy repo correctly?")
        return

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    written = 0

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        for project_name in sorted(os.listdir(projects_dir)):
            bugs_dir = os.path.join(projects_dir, project_name, "bugs")
            if not os.path.isdir(bugs_dir):
                continue

            for bug_id in sorted(os.listdir(bugs_dir)):
                bug_folder = os.path.join(bugs_dir, bug_id)
                patch_path = os.path.join(bug_folder, "bug_patch.txt")
                info_path = os.path.join(bug_folder, "bug_info.txt")
                if not os.path.exists(patch_path):
                    continue

                with open(patch_path, "r", encoding="utf-8", errors="ignore") as f:
                    patch_text = f.read()

                buggy_code, fixed_code = split_patch_into_before_after(patch_text)
                if not buggy_code and not fixed_code:
                    continue

                info = parse_bug_info(info_path)

                entry = NormalizedBug(
                    id=f"bugsinpy_{project_name}_{bug_id}",
                    dataset_source="BugsInPy",
                    language="python",
                    buggy_code=buggy_code,
                    fixed_code=fixed_code,
                    error=None,
                    bug_type=None,
                    bug_description=info.get("bug_report") or info.get("BugReport") or None,
                    solution=None,
                    repository=project_name,
                    file=None,
                    metadata={"bug_id": bug_id, **info},
                )
                out.write(json.dumps(entry.to_dict()) + "\n")
                written += 1

    print(f"Wrote {written} normalized entries to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

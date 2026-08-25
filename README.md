# RAG-Enhanced LLM for GitHub Bug Detection and Recovery

A mini project that scans a public GitHub repository, finds candidate bugs
using real static analysis, retrieves similar historical bugs/fixes from a
knowledge base (BugsInPy, Bugs2Fix, RunBugRun) using RAG, and asks an LLM
to explain each finding and suggest a fix — or say plainly when it isn't
confident enough to.

```
GitHub URL -> Validate -> Download repo -> Scan files -> Detect candidates
-> RAG retrieve similar historical bugs -> LLM analyze -> Bug report UI
```

## What's real vs. what needs one extra step

Everything in this project runs and has been tested — the AST-based bug
detector, the FastAPI backend, the FAISS retrieval mechanics, the frontend,
and the no-API-key fallback path. Two things need internet access on
**your** machine (not available while this was built) before your first
real run:

1. **Downloading the embedding model** (`all-MiniLM-L6-v2`, ~90MB, from
   Hugging Face) — happens automatically the first time you run the app.
2. **Downloading the three real datasets** — a bundled 8-entry sample
   dataset (`datasets/normalized/sample.jsonl`) is included so the RAG
   pipeline works out of the box. Run the scripts in step 4 below to swap
   in the real datasets before your demo.

## 1. Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

## 2. Set your API key

```bash
cp .env.example .env
# then open .env and paste your Anthropic API key
```

Without a key, the app still runs — it just shows the raw static-analysis
finding without LLM-generated explanations/fixes (clearly labeled in the
UI as "No LLM API key configured").

## 3. (Optional but recommended) Build the real datasets

By default the app uses the small bundled sample dataset. To use the real
three datasets from the abstract:

```bash
cd datasets
pip install datasets   # Hugging Face datasets library

# Bugs2Fix (CodeXGLUE code-refinement task)
python normalize_bugs2fix.py

# RunBugRun (keeps Python + Java, capped at 2000 each — edit the script to change)
python normalize_runbugrun.py

# BugsInPy (clone the repo first, it's not on Hugging Face)
git clone https://github.com/soarsmu/BugsInPy.git
python normalize_bugsinpy.py --bugsinpy-path ./BugsInPy
```

Each script writes a `.jsonl` file into `datasets/normalized/`. You can
keep the sample file alongside the real ones, remove it, or keep just one
dataset — the index builder picks up every `.jsonl` file in that folder.

## 4. Build the vector index

```bash
cd backend
python -m rag.build_index
```

This downloads the embedding model on first run, embeds every record, and
writes `backend/rag/index/faiss.index` + `metadata.json`. Re-run this any
time you change the normalized datasets. (If you skip this step, the
backend builds it automatically on first scan request — but doing it
ahead of time means your demo doesn't stall on the first click.)

## 5. Start the app

```bash
cd backend
uvicorn main:app --reload --port 8000
```

Open **http://localhost:8000** in your browser — the backend also serves
the frontend, so there's nothing separate to start.

## 6. Try it

Paste a public repo URL and click **Scan Repository**, for example:

```
https://github.com/psf/requests
```

You'll see: validation, a progress line, a summary (files scanned / bugs
found / severity breakdown), and a bug card per finding with location,
cause, current code, suggested replacement, RAG evidence from the
knowledge base, and a confidence score.

## Project structure

```
project/
├── backend/
│   ├── main.py                 # FastAPI app - orchestrates the whole flow
│   ├── models.py                # Shared Pydantic schemas
│   ├── github_handler.py        # URL validation + repo download
│   ├── file_scanner.py          # Recursive source file discovery
│   ├── llm_client.py            # Calls the LLM with RAG context
│   ├── detectors/
│   │   └── python_detector.py   # Real AST-based bug detector
│   ├── rag/
│   │   ├── build_index.py       # Builds the FAISS index
│   │   ├── retriever.py         # Similarity search at scan time
│   │   └── index/                # Generated - faiss.index + metadata.json
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   └── index.html               # Single-file UI, no build step required
├── datasets/
│   ├── schema.py                 # Normalized record shape
│   ├── normalize_bugs2fix.py
│   ├── normalize_runbugrun.py
│   ├── normalize_bugsinpy.py
│   └── normalized/
│       └── sample.jsonl          # Bundled so the app runs out of the box
└── README.md
```

## Honest limitations (say these in your viva, don't hide them)

- Only Python has a real static-analysis detector in this build (the
  file scanner recognizes JS/TS/Java/C/C++/C#/Go/PHP files for the
  "scan the whole repo" requirement, but `detectors/` currently only has
  `python_detector.py`). Adding another language means writing a new
  detector file — the plumbing in `main.py` already supports it via the
  `DETECTORS_BY_EXTENSION` dict.
- The detector uses real AST analysis for a fixed set of patterns
  (division by zero, bare except, mutable default args, `== None`,
  possibly-unused functions). It is not a general-purpose bug finder —
  no static analyzer is. This matches the abstract's own point: report
  "possible" bugs with a confidence score, not certainties.
- "Possibly unused function" is a same-file heuristic — it can't see
  whether another file imports and calls that function, so it's
  reported at lower confidence for exactly that reason.
- The RAG knowledge base quality depends entirely on which datasets you
  build it from. The bundled sample has 8 hand-written entries just to
  prove the pipeline works — build the real datasets (step 3) before
  your actual demo/evaluation.

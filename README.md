# RAG-Enhanced LLM for GitHub Bug Detection and Recovery

A full-stack AI-based system that scans a GitHub repository, detects potential bugs using static analysis, retrieves similar historical bugs using a RAG pipeline, and uses a Large Language Model (LLM) to explain the detected issue and suggest a possible fix.

## Project Overview

The system follows this workflow:

```text
GitHub Repository URL
        ↓
Repository Validation
        ↓
Download Repository
        ↓
Source File Scanning
        ↓
Static Bug Detection
        ↓
RAG Retrieval
        ↓
LLM Analysis
        ↓
Bug Explanation + Suggested Fix
        ↓
Frontend Bug Report
```

The application supports source files such as:

- Python
- JavaScript
- JSX
- TypeScript
- TSX
- Java
- C
- C++
- C#
- Go
- PHP

Python files receive AST-based checks plus Pyflakes static analysis, including
syntax issues, undefined names, and division-by-zero risks. Each detected
finding is sent to the LLM separately for a code-specific explanation. If the
LLM is unavailable or its usage limit is reached, the application reports that
status instead of presenting a generated cause or solution.

---

# 1. Requirements

Before running the project locally, install the following:

### Required

- Windows 10/11
- VS Code
- Python **3.11.x**
- Git
- Internet connection
- A modern web browser

### Recommended Python Version

Use:

```text
Python 3.11.9
```

Python 3.14 is **not recommended** for this project because some dependencies used by the project are pinned to versions designed for the Python 3.11 environment.

Check your Python version:

```powershell
python --version
```

If Python 3.11 is installed through the Python launcher, check it using:

```powershell
py -3.11 --version
```

---

# 2. Download the Project

Download or clone the repository.

If using Git:

```powershell
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

Then open the project folder in VS Code.

The project structure should look similar to:

```text
Git-bug-detection-main/
│
├── backend/
│   ├── detectors/
│   ├── rag/
│   ├── test/
│   ├── .env.example
│   ├── file_scanner.py
│   ├── github_handler.py
│   ├── llm_client.py
│   ├── main.py
│   ├── models.py
│   ├── requirements.txt
│   └── runtime.txt
│
├── datasets/
│   ├── normalized/
│   ├── normalize_bugs2fix.py
│   ├── normalize_bugsinpy.py
│   ├── normalize_runbugrun.py
│   └── schema.py
│
├── frontend/
│   ├── index.html
│   └── icon1.png
│
└── README.md
```

---

# 3. Open the Project in VS Code

Open VS Code and select:

```text
File → Open Folder
```

Select:

```text
Git-bug-detection-main
```

Then open a new VS Code terminal:

```text
Terminal → New Terminal
```

Make sure the terminal is inside the project root:

```text
...\Git-bug-detection-main>
```

---

# 4. Create a Python Virtual Environment

Create a virtual environment using Python 3.11:

```powershell
py -3.11 -m venv venv
```

Activate it:

```powershell
.\venv\Scripts\Activate.ps1
```

After activation, the terminal should show:

```text
(venv) PS C:\...\Git-bug-detection-main>
```

If you see `(venv)` at the beginning of the terminal, the virtual environment is active.

---

# 5. Install Backend Dependencies

From the project root, run:

```powershell
pip install -r backend\requirements.txt
```

The project uses packages including:

- FastAPI
- Uvicorn
- Requests
- Pydantic
- Python-dotenv
- GitPython
- Scikit-learn
- Joblib
- NumPy
- Google Generative AI

Wait until the installation finishes successfully.

---

# 6. Configure the Gemini API Key

The LLM component can use Google Gemini to analyze detected bugs and generate explanations and fixes.

Go into the backend folder:

```powershell
cd backend
```

Copy the example environment file:

```powershell
Copy-Item .env.example .env
```

Open:

```text
backend/.env
```

Add your Gemini API key:

```text
GEMINI_API_KEY=your_gemini_api_key_here
```

You can obtain a Gemini API key from:

```text
https://aistudio.google.com/
```

The application supports up to 10 Gemini keys and 5 Groq keys. Set the
numbered variables in `backend/.env`:

```text
GEMINI_API_KEY_1=key1
GEMINI_API_KEY_2=key2
...
GEMINI_API_KEY_10=key10
GROQ_API_KEY_1=key1
...
GROQ_API_KEY_5=key5
```

For each finding, the backend tries the configured Gemini keys, then the
configured Groq keys if Gemini cannot complete the analysis. Only one
successful explanation is used per finding. Providers still enforce their own
rate limits and account/project quotas; extra keys cannot make usage unlimited.
The backend loads `backend/.env` even when started from another working folder.

Do **not** commit your `.env` file or API keys to GitHub.

---

# 7. RAG Knowledge Base

The project contains a RAG pipeline for retrieving similar historical bugs.

The normalized dataset directory is:

```text
datasets/normalized/
```

A sample dataset is included with the project so that the RAG pipeline can run without downloading the complete external datasets.

The RAG component uses:

```text
TF-IDF
```

with Scikit-learn.

It does not require a large neural embedding model or PyTorch installation.

---

# 8. Optional: Build the RAG Index

The application can build the RAG index automatically when required.

You can also build it manually.

From the `backend` folder:

```powershell
python -m rag.build_index
```

This generates the required RAG index files.

Run this again whenever the normalized datasets are changed.

---

# 9. Start the Backend

Make sure you are inside:

```text
backend
```

Your terminal should look similar to:

```text
(venv) PS C:\...\Git-bug-detection-main\backend>
```

Start FastAPI using:

```powershell
python -m uvicorn main:app --reload
```

If the server starts successfully, you should see:

```text
Uvicorn running on http://127.0.0.1:8000
```

and:

```text
Application startup complete.
```

The backend is now running locally.

---

# 10. Open the Application

Open a browser and visit:

```text
http://127.0.0.1:8000
```

or:

```text
http://localhost:8000
```

The FastAPI backend serves the frontend directly, so a separate frontend development server is **not required** for the current project.

The frontend is located in:

```text
frontend/index.html
```

and FastAPI serves it automatically when the backend is running.

---

# 11. Check Backend Health

You can check whether the backend is working by opening:

```text
http://127.0.0.1:8000/api/health
```

A successful response should look like:

```json
{
  "status": "ok"
}
```

---

# 12. Scan a GitHub Repository

After opening the application:

1. Enter a **public GitHub repository URL**.
2. Click the scan button.
3. The application validates the repository.
4. The repository source files are downloaded temporarily.
5. Supported source files are identified.
6. Static detectors analyze the source code.
7. Similar historical bugs are retrieved from the RAG knowledge base.
8. The LLM analyzes the findings.
9. The application generates the bug report.

The result contains information such as:

- Bug type
- Error description
- File name
- Function
- Line number
- Cause
- Why the problem occurs
- Current code
- Suggested replacement code
- Explanation
- Confidence level
- Similar historical bugs retrieved from the RAG system

---

# 13. Example Backend Output

During a scan, the terminal may show output similar to:

```text
[github] Selected 82 source files
[github] Downloaded 82 source files
[scanner] Found 82 supported source files
[scan] Files discovered: 82
[scan] Findings detected: 929
```

This indicates that the repository was successfully downloaded and analyzed.

The exact number of files and findings will depend on the repository being scanned.

---

# 14. Important: Keep the Backend Terminal Running

Once you run:

```powershell
python -m uvicorn main:app --reload
```

do **not** close the terminal.

The backend must remain running while you use the website.

To stop the server, press:

```text
Ctrl + C
```

---

# 15. Running the Project Again Later

After restarting your computer or VS Code, you do not need to recreate the virtual environment.

Open the project in VS Code and activate the existing environment:

```powershell
.\venv\Scripts\Activate.ps1
```

Then:

```powershell
cd backend
```

Start the application:

```powershell
python -m uvicorn main:app --reload
```

Then open:

```text
http://127.0.0.1:8000
```

---

# 16. Complete Windows Setup — Quick Version

For a fresh installation, the complete command sequence is:

```powershell
py -3.11 -m venv venv
```

```powershell
.\venv\Scripts\Activate.ps1
```

```powershell
pip install -r backend\requirements.txt
```

```powershell
cd backend
```

Create/configure:

```text
.env
```

Then start:

```powershell
python -m uvicorn main:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

---

# 17. Troubleshooting

## Python 3.11 not found

Run:

```powershell
py install 3.11
```

Then check:

```powershell
py -3.11 --version
```

---

## Virtual environment activation fails

Try:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then:

```powershell
.\venv\Scripts\Activate.ps1
```

---

## `uvicorn` command not recognized

Use:

```powershell
python -m uvicorn main:app --reload
```

instead of:

```powershell
uvicorn main:app --reload
```

---

## Port 8000 is already in use

Start the application on another port:

```powershell
python -m uvicorn main:app --reload --port 8001
```

Then open:

```text
http://127.0.0.1:8001
```

---

## Gemini API key is not working

Check that:

```text
backend/.env
```

contains:

```text
GEMINI_API_KEY=your_actual_key
```

Make sure there are no unnecessary quotes or spaces.

Restart the backend after changing `.env`.

---

## Application runs without an LLM API key

Static detection still runs, but causes and solutions require the LLM. When
the LLM is unavailable or reaches its usage limit, each finding displays:

```text
LLM limit completed / unavailable
The cause and solution could not be generated because the LLM is currently unavailable or its usage limit has been reached. Please try again later.
```

The application does not substitute a generic cause or solution in this case.
Configure a valid Gemini API key to receive per-finding explanations.

---

# 18. Project Architecture

```text
                    ┌─────────────────────┐
                    │   GitHub Repository │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Repository Handler  │
                    │ Validation/Download │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   File Scanner      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Static Bug Detectors│
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    RAG Retriever    │
                    │ TF-IDF Knowledge    │
                    │      Base           │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     Gemini LLM      │
                    │ Bug Analysis + Fix  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Bug Report UI    │
                    └─────────────────────┘
```

---

# 19. Project Structure

```text
Git-bug-detection-main/
│
├── backend/
│   ├── detectors/
│   │   ├── python_detector.py
│   │   ├── js_detector.py
│   │   ├── cfamily_detector.py
│   │   └── syntax_balance.py
│   │
│   ├── rag/
│   │   ├── build_index.py
│   │   └── retriever.py
│   │
│   ├── test/
│   │   └── test_syntax_repairs.py
│   │
│   ├── .env.example
│   ├── file_scanner.py
│   ├── github_handler.py
│   ├── llm_client.py
│   ├── main.py
│   ├── models.py
│   ├── requirements.txt
│   └── runtime.txt
│
├── datasets/
│   ├── normalized/
│   │   ├── common_syntax_bugs.jsonl
│   │   └── sample.jsonl
│   ├── normalize_bugs2fix.py
│   ├── normalize_bugsinpy.py
│   ├── normalize_runbugrun.py
│   └── schema.py
│
├── frontend/
│   ├── index.html
│   └── icon1.png
│
└── README.md
```

---

# 20. Current System Limitations

This project is a static-analysis and RAG-assisted bug detection system. It is not a complete replacement for a compiler, IDE, or production-grade static analyzer.

Important limitations include:

- Not every bug in a repository can be detected.
- Python uses AST-based detection for supported patterns.
- Other supported languages use structural and heuristic checks.
- Some findings may be false positives.
- Same-file analysis cannot always determine whether a function is used from another file.
- TF-IDF retrieval depends on vocabulary overlap and is not semantic neural embedding retrieval.
- The bundled sample dataset is small and is intended mainly to demonstrate the RAG pipeline.
- LLM output depends on the configured Gemini model/API availability.
- Very large repositories may take longer to scan and analyze.

Therefore, findings should be treated as **potential bugs**, and the suggested fixes should be reviewed before being applied to production code.

---

# 21. Technologies Used

### Frontend

- HTML
- CSS
- JavaScript

### Backend

- Python
- FastAPI
- Uvicorn

### Code Analysis

- Python AST
- Structural analysis
- Pattern-based heuristics

### RAG

- Scikit-learn
- TF-IDF
- Joblib
- JSONL knowledge base

### LLM

- Google Gemini API

### Repository Processing

- GitPython
- GitHub repository source downloading

---

# 22. Local Execution Summary

The simplest way to remember the execution process is:

```text
1. Install Python 3.11
          ↓
2. Open project in VS Code
          ↓
3. Create virtual environment
          ↓
4. Activate virtual environment
          ↓
5. Install requirements
          ↓
6. Configure .env
          ↓
7. Start FastAPI
          ↓
8. Open localhost:8000
          ↓
9. Enter GitHub repository URL
          ↓
10. Scan repository
```

### Main command

```powershell
python -m uvicorn main:app --reload
```

### Local application

```text
http://127.0.0.1:8000
```

### Health check

```text
http://127.0.0.1:8000/api/health
```

---

# 23. Author / Project

**Project:** RAG-Enhanced LLM for GitHub Bug Detection and Recovery

**Purpose:** Detect potential bugs in GitHub repositories, retrieve related historical bug information, and use an LLM to provide explanations and suggested code fixes.
# Auto Grading System

An **Automated Code Grading System** that uses Control Flow Graph (CFG) comparison, sandbox execution, and Google Gemini LLM integration to automatically grade Python code submissions and provide detailed feedback.

---

## Architecture Overview

```
Input: Question + Reference Solution + Student Code
        │
        ▼
┌────────────────────────────────────────────────────────┐
│                   Graph Pipeline                       │
│  Build CFG (Reference)  ──┐                            │
│  Build CFG (Student)    ──┴──► Graph Comparison        │
│                              & Difference Detection    │
│                              │                         │
│                              ▼                         │
│                      Error Localization &              │
│                      Severity Classification           │
│                              │                         │
│                              ▼                         │
│                      Step-by-Step Repair Generation    │
└────────────────────────────────────────────────────────┘
        │
        ├──► Sandbox Executor (test case results)
        │
        └──► LLM (Gemini 2.0 Flash) — natural language feedback
                              │
                              ▼
            Score + Detailed Feedback + Repair Guide
```

---

## Project Structure

```
Auto_Grading_System/
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI entrypoint
│   ├── config.py                # Settings from environment variables
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py            # API endpoints
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py           # Pydantic request/response models
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── builder.py           # CFG builder using ast module
│   │   └── comparator.py        # Graph comparison & diff detection
│   ├── sandbox/
│   │   ├── __init__.py
│   │   └── executor.py          # Safe subprocess code executor
│   ├── llm/
│   │   ├── __init__.py
│   │   └── gemini_client.py     # Gemini 2.0 Flash integration
│   ├── scoring/
│   │   ├── __init__.py
│   │   └── engine.py            # Scoring engine
│   ├── repair/
│   │   ├── __init__.py
│   │   └── generator.py         # Repair guide generator
│   └── error/
│       ├── __init__.py
│       └── classifier.py        # Error localization & classification
├── tests/
│   ├── __init__.py
│   ├── test_graph.py
│   ├── test_sandbox.py
│   ├── test_scoring.py
│   └── test_api.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.11+
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/pminh05/Auto_Grading_System.git
cd Auto_Grading_System

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env and set your GEMINI_API_KEY
```

---

## Environment Setup

Copy `.env.example` to `.env` and fill in the values:

```env
GEMINI_API_KEY=your_gemini_api_key_here
MAX_EXECUTION_TIME=5
MAX_MEMORY_MB=256
DEBUG=false
```

| Variable | Description | Default |
|---|---|---|
| `GEMINI_API_KEY` | Google Gemini API key | *(required)* |
| `MAX_EXECUTION_TIME` | Sandbox timeout in seconds | `5` |
| `MAX_MEMORY_MB` | Max memory per execution in MiB | `256` |
| `DEBUG` | Enable debug logging | `false` |

---

## Running the Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The interactive API documentation is available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## API Documentation

### `POST /api/v1/grade`

Grade a student code submission against a reference solution.

**Request:**
```json
{
  "question": "Write a function that sums a list.",
  "reference_solution": "def sum_list(lst):\n    return sum(lst)\n",
  "student_code": "def sum_list(lst):\n    total = 0\n    for x in lst:\n        total += x\n    return total\n",
  "test_cases": [
    { "input": "", "expected_output": "" }
  ]
}
```

**Response:**
```json
{
  "score": {
    "total_score": 85.0,
    "graph_similarity_score": 30.0,
    "test_pass_score": 40.0,
    "code_quality_score": 17.0,
    "deductions": 0.0,
    "breakdown": {}
  },
  "feedback": "...",
  "repair_guide": { "steps": [], "summary": "..." },
  "execution_result": { "passed": 1, "failed": 0, ... },
  "graph_diff": { "similarity_score": 0.75, ... },
  "errors": []
}
```

---

### `POST /api/v1/execute`

Execute code against test cases in the sandbox.

**Request:**
```json
{
  "code": "print(int(input()) * 2)",
  "test_cases": [
    { "input": "5", "expected_output": "10" }
  ]
}
```

---

### `POST /api/v1/analyze`

Compare two code submissions at the graph level without full grading.

**Request:**
```json
{
  "reference_code": "for i in range(10):\n    print(i)\n",
  "student_code": "i = 0\nwhile i < 10:\n    print(i)\n    i += 1\n"
}
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Module Descriptions

| Module | Description |
|---|---|
| `app/graph/builder.py` | Parses Python source with `ast` and builds a directed CFG. Each node represents one operation; each edge is a control-flow transition. |
| `app/graph/comparator.py` | Compares two CFGs and reports missing/extra nodes and edges plus a Jaccard-based similarity score. |
| `app/error/classifier.py` | Maps graph differences to typed, severity-rated errors (SYNTAX, LOOP, ALGORITHM, DATA_HANDLING × CRITICAL/MAJOR/MINOR). |
| `app/sandbox/executor.py` | Runs student code in a sandboxed subprocess with configurable timeout and captures stdout/stderr per test case. |
| `app/llm/gemini_client.py` | Wraps the Google Generative AI SDK to call Gemini 2.0 Flash for natural-language feedback, summaries, and repair hints. |
| `app/scoring/engine.py` | Computes a weighted 0–100 score from graph similarity (40%), test pass rate (40%), and code quality (20%) with per-severity deductions. |
| `app/repair/generator.py` | Generates an ordered, step-by-step repair guide by combining error classification with LLM-generated hints. |
| `app/api/routes.py` | FastAPI route handlers that orchestrate the full grading pipeline. |

---

## Quick Start: Batch Grading CLC11

```bash
pip install -r requirements.txt
python scripts/grade_CLC11.py
# Output: output/CLC11_results.csv, output/CLC11_results.json
```

---

## License

MIT
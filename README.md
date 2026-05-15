# KIWI-8 AI Onboarding Chatbot

KIWI-8 is a RAG-based AI onboarding chatbot that answers employee HR and onboarding questions from company PDF documents.

The app is built for fast internal self-service across topics such as holidays, onboarding, VPN, payroll, reimbursement, insurance, cafeteria, and policy-related queries.

## Problem Statement

Employees often need quick answers from multiple HR and onboarding documents. Searching PDFs manually is slow, inconsistent, and dependent on knowing the exact document or section name.

KIWI-8 solves this by allowing users to ask natural-language questions and receive short, relevant answers grounded in uploaded PDF content.

## Business Value

- Reduces repetitive HR and onboarding support questions.
- Helps new joiners find policy and process answers faster.
- Provides a single searchable interface over multiple PDF documents.
- Improves response consistency by grounding answers in source material.
- Supports 24x7 self-service for common employee queries.

## Features

- Streamlit web interface
- PDF-based knowledge retrieval
- Text chunking with overlap
- FAISS semantic search
- BM25 keyword search
- Cross-encoder reranking for better relevance
- Groq LLM answer generation
- Answer cleanup for navigation/reference text
- Local PDF folder support
- Environment-based API key loading

## Tech Stack

- Python
- Streamlit
- LangChain
- FAISS
- BM25
- HuggingFace embeddings
- SentenceTransformers cross-encoder reranker
- Groq LLM
- PyPDF

## Architecture

```text
User Question
     |
     v
Streamlit UI
     |
     v
PDF Text Extraction
     |
     v
Text Chunking
     |
     v
FAISS Semantic Search + BM25 Keyword Search
     |
     v
Candidate Chunk Collection
     |
     v
Cross-Encoder Reranking
     |
     v
Top Context Chunks
     |
     v
Groq LLM
     |
     v
Clean Final Answer
```

## Repository Structure

```text
ChatBot/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .env.example
├── pdfs/
│   └── add-your-pdfs-here.pdf
└── faiss_index/        # generated locally, ignored by Git
```

## Local Setup

Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Set your Groq API key:

```powershell
$env:GROQ_API_KEY="your_groq_api_key"
```

Or create a local `.env` / Streamlit secrets setup if your environment supports it.

## Add Documents

Place PDFs inside:

```text
pdfs/
```

Example:

```text
pdfs/HOLIDAYS PUNE5.pdf
```

The app reads PDF text, splits it into chunks, builds a FAISS index, and uses BM25 plus reranking for retrieval.

## Run Locally

```powershell
streamlit run app.py
```

Open the URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Deployment Notes

The event guidance says deployment should use a Dockerfile and expose port `8000`.

For Streamlit deployment, the container should start the app like this:

```bash
streamlit run app.py --server.port=8000 --server.address=0.0.0.0
```

Before deployment, verify:

- `GROQ_API_KEY` or the provided LLM key is available as an environment variable.
- PDFs required for the demo are included or mounted correctly.
- The app starts without relying on local-only files.
- The final submission does not include `venv/`, `.cache/`, or generated index folders.

## Environment Variables

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY` | API key used by the Groq LLM integration |

If the event provides a different LLM provider or key, update the LLM loading logic in `app.py`.

## Evaluation Questions

Use these questions to test the chatbot:

```text
What are the Pune holidays?
How do I connect to VPN?
How do I claim reimbursement?
What is the payroll process?
What insurance benefits are available?
How to complain?
How to reset laptop password?
```

Expected behavior:

- The answer should be based only on the PDF content.
- If the answer is not present, the app should ask the user to reach out to HR.
- The chatbot should avoid irrelevant matches caused by single keywords.
- The reranker should prefer chunks that answer the full question.

## Known Limitations

- PDF table extraction may not be perfect for complex tables.
- First run can be slower because embedding and reranker models may download.
- FAISS index is generated locally and should not be pushed to Git.
- The current app uses local PDFs, not a managed document store.
- The current app does not include authentication or user-level document permissions.

## Repository Hygiene

Do not push local runtime or generated dependency folders.

The `.gitignore` should include:

```gitignore
.streamlit/secrets.toml
faiss_index/
.cache/
__pycache__/
.env
venv/
*.pyc
app.py.bak
```

Use `requirements.txt` to recreate dependencies instead of submitting the local virtual environment.

## Submission Checklist

- [ ] Source code is committed.
- [ ] `requirements.txt` is present.
- [ ] README explains setup, API key, run steps, and architecture.
- [ ] Dockerfile is added for deployment.
- [ ] App runs on port `8000` in deployment.
- [ ] Demo PDFs are available for evaluation.
- [ ] Demo video is added if required.
- [ ] Presentation deck is added if required.
- [ ] `venv/`, `.cache/`, and generated indexes are not submitted.
- [ ] App is tested in a fresh environment.

## Demo Script

Suggested 5-7 minute demo flow:

1. Explain the employee onboarding problem.
2. Show the PDF document folder.
3. Start the Streamlit app.
4. Ask a direct HR question.
5. Ask a query with tricky wording, such as `How to reset laptop password?`.
6. Explain that FAISS, BM25, and reranking improve answer relevance.
7. Show how the final answer is short and grounded in the uploaded documents.
8. Close with business value and deployment readiness.

## Final Notes

KIWI-8 is designed as a simple, practical RAG prototype. The strongest improvement areas before final submission are Docker deployment, README completeness, demo video, presentation deck, and testing in a clean environment.

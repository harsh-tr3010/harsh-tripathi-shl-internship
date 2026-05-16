# SHL Assessment Recommender API

A production-grade, stateless FastAPI microservice that guides users from a vague hiring intent to a grounded shortlist of SHL individual test solutions through dynamic, multi-turn dialogue.

## Author
* **Name:** Harsh Tripathi
* **Email:** harshtripathi803@gmail.com

---

## System Architecture

The application is engineered as a deterministic state machine built on a stateless REST architecture. Every interaction cycle handles the entire conversation history to evaluate transitions across distinct conversational stages: Clarification, Direct Recommendation, Shortlist Refinement, Narrative Comparison, and Session Termination.

### Core Pipeline Execution Flow
1. **Request Ingestion:** The client transmits a structured payload containing the stateless text transcript of the conversation thread.
2. **Context-Aware Retrieval (RAG):** The search engine extracts keyword features from the aggregated transcript and scores them against the localized catalog index. The corpus is dynamically compressed down to the top 15–18 hyper-focused assessment vectors before LLM execution, bypassing upstream rate limits while maximizing context alignment.
3. **Deterministic Inference:** The payload is processed using `Llama 3.3 70B` via the Groq inference pipeline. Decodings are restricted to `temperature=0.0` to preserve structural layout constraints and isolate logic flags from random variance.
4. **Defensive Structural Parsing:** An internal post-processing layer sanitizes the model output to normalize structural key drifts before sending the response back to the client.

---

## Directory Structure

```text
├── data/
│   └── catalog.json          # Pre-computed and cleaned individual test solutions
├── .env                      # Local infrastructure environment keys
├── agent.py                  # Context-engineering engine, RAG module, and inference call
├── fetch_data.py             # Data ingestion pipeline and clean filtering mechanics
├── main.py                   # FastAPI application initialization and routing setup
├── models.py                 # Pydantic schemas enforcing structural compliance
└── requirements.txt          # Explicit package version lockfile
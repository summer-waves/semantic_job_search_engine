# Job Search Engine

Semantic job search API built with FAISS + BM25 over 43,484 real job postings.
Designed to mirror the retrieval architecture used by companies like Indeed and LinkedIn.

## What this demonstrates

- Building a vector search pipeline from scratch (no LangChain, no shortcuts)
- Comparing retrieval methods empirically (FAISS vs BM25 ablation)
- Serving ML models via a production-ready REST API
- Writing tests against a live API with pytest

## Architecture
User Query
↓
TF-IDF Embedder (scikit-learn)
↓
FAISS IndexFlatIP (cosine similarity over 43K vectors)
↓
FastAPI /search endpoint
↓
Ranked JSON results

BM25 keyword search available at `/search/bm25` for comparison.

## Stack

| Layer | Tool |
|---|---|
| Vector search | FAISS (IndexFlatIP) |
| Keyword search | BM25 (rank-bm25) |
| Embeddings | TF-IDF (scikit-learn) |
| API | FastAPI + uvicorn |
| Data | pandas + pyarrow |
| Tests | pytest |
| Container | Docker |

## Ablation: FAISS vs BM25

![FAISS vs BM25 comparison](notebooks/faiss_vs_bm25.png)

**Finding:** BM25 outperforms on exact keyword queries (e.g. "python developer fintech").
FAISS outperforms on semantic queries (e.g. "business intelligence analyst").
Production search systems like Indeed use hybrid approaches combining both.

## Run locally

```bash
# Install dependencies
pip install -r requirements.txt

# Start the API
uvicorn app.main:app --reload --port 8000
```

## API endpoints

| Endpoint | Description |
|---|---|
| `GET /` | Health check — returns total jobs indexed |
| `GET /search?q=...&top_k=5` | Semantic search via FAISS |
| `GET /search/bm25?q=...&top_k=5` | Keyword search via BM25 |
| `GET /docs` | Interactive Swagger UI |

## Example response

```json
{
  "query": "data analyst remote",
  "total_results": 5,
  "results": [
    {
      "title": "Data Analyst (remote)",
      "company": "Ad Hoc",
      "score": 1.0
    }
  ]
}
```

## Tests

```bash
pytest tests/ -v
```

4 tests covering: API health, search results, pagination, and score validity.

## Dataset

43,484 job postings sourced from Kaggle
(lukebarousse/data-analyst-job-postings-google-search).
Raw data excluded from this repo per Kaggle's redistribution terms.

## Note on embeddings

This project uses TF-IDF vectors as a network-accessible fallback.
The architecture is designed to swap in `sentence-transformers/all-MiniLM-L6-v2`
for dense semantic embeddings — one line change in `app/embedder.py`.
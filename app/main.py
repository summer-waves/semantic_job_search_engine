import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, Query
from app.retriever import Retriever

retriever = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever
    print("Loading job data and building index...")
    df = pd.read_parquet("data/cleaned_jobs.parquet")
    retriever = Retriever(df)
    print("Ready.")
    yield

app = FastAPI(
    title="Job Search Engine",
    description="Semantic job search using sentence-transformers + FAISS",
    lifespan=lifespan
)

@app.get("/")
def root():
    return {"status": "running", "total_jobs": retriever.index.ntotal}

@app.get("/search")
def search(
    q: str = Query(..., description="Search query"),
    top_k: int = Query(10, ge=1, le=50)
):
    results = retriever.search(q, top_k)
    return {
        "query": q,
        "total_results": len(results),
        "results": results
    }

@app.get("/search/bm25")
def search_bm25(
    q: str = Query(..., description="Search query"),
    top_k: int = Query(10, ge=1, le=50)
):
    results = retriever.search_bm25(q, top_k)
    return {
        "query": q,
        "method": "bm25",
        "total_results": len(results),
        "results": results
    }
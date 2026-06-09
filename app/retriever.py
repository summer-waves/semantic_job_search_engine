import faiss
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from app.embedder import embedder


class Retriever:
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

        # Build FAISS index
        print(f"Encoding {len(df)} job postings...")
        texts = self.df["search_text"].tolist()
        embedder.fit(texts)
        embeddings = embedder.encode(texts)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        print(f"Index built with {self.index.ntotal} vectors.")

        # Build BM25 index (tokenize by whitespace)
        print("Building BM25 index...")
        tokenized = [t.lower().split() for t in texts]
        self.bm25 = BM25Okapi(tokenized)
        print("BM25 ready.")

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        query_vec = embedder.encode([query])
        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            row = self.df.iloc[idx]
            results.append({
                "title": row.get("job_title", "N/A"),
                "company": row.get("company_name", "N/A"),
                "description": str(row.get("job_description", ""))[:300],
                "score": round(float(score), 4)
            })
        return results

    def search_bm25(self, query: str, top_k: int = 10) -> list[dict]:
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            row = self.df.iloc[idx]
            results.append({
                "title": row.get("job_title", "N/A"),
                "company": row.get("company_name", "N/A"),
                "description": str(row.get("job_description", ""))[:300],
                "score": round(float(scores[idx]), 4)
            })
        return results


if __name__ == "__main__":
    df = pd.read_parquet("data/cleaned_jobs.parquet")
    r = Retriever(df)
    print("\n--- FAISS results ---")
    for res in r.search("machine learning engineer remote", top_k=3):
        print(res)
    print("\n--- BM25 results ---")
    for res in r.search_bm25("machine learning engineer remote", top_k=3):
        print(res)
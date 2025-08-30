# app/api.py
from fastapi import FastAPI
from index.search_engine import SearchEngine

app = FastAPI()
engine = SearchEngine()

@app.post("/search")
def search(query: str, top_k: int = 5):
    results = engine.search(query, top_k=top_k)
    return {"query": query, "results": results}
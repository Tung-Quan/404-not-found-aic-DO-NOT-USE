# run_demo.py
from index.search_engine import SearchEngine

engine = SearchEngine()

query = "người đàn ông đội mũ"
results = engine.search(query, top_k=5)

print("Query:", query)
for r in results:
    print(r)
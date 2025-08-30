Single-model CLIP search notebook

Files:
- single_model_rerank_system.ipynb: interactive notebook (indexing + retrieval + rerank).
- src/single_model_index.py: reusable module extracted from the notebook.
- scripts/build_single_model_index.py: CLI script to build embeddings and FAISS index.

Quick run (from project root):

# install deps (if using conda/venv)
# pip install -r requirements.txt  # ensure transformers, torch, faiss-cpu or faiss-gpu

# Run the CLI builder (downloads model, encodes frames under ./frames)
python scripts/build_single_model_index.py --rebuild

# Then you can open the notebook and run the retrieval demo cell.

Notes:
- If model loading hits HF gated repos, set HUGGINGFACE_HUB_TOKEN in your environment.
- For constrained GPUs, reduce batch size and consider CPU fallback.
- Metadata (paths_metadata.json) maps frame path -> pts_time when CSV mapping exists.

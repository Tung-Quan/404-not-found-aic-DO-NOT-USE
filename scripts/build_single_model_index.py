"""CLI helper to build the single-model CLIP index.
Usage:
    python scripts/build_single_model_index.py --rebuild
"""
import argparse
from src.single_model_index import CLIPEncoder, build_embeddings_and_index

parser = argparse.ArgumentParser()
parser.add_argument('--model', default=None)
parser.add_argument('--rebuild', action='store_true')
args = parser.parse_args()

model = args.model or None
enc = CLIPEncoder(model_name=model) if model else CLIPEncoder()
E, paths = build_embeddings_and_index(enc, rebuild=args.rebuild)
print('Done. embeddings shape:', getattr(E, 'shape', None), 'num paths:', len(paths))

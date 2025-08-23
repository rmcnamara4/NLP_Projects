import argparse
import json 
import os 

from src.retrieval.pipeline import load_index, load_embedder, run_search
from src.utils.io import load_jsonl, save_jsonl

def main():     
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '--queries_path', 
        default = 'data/queries.jsonl'
    )
    ap.add_argument(
        '--index_path', 
        default = 'index/faiss_faiss_store'
    )
    ap.add_argument(
        '--k', 
        type = int, 
        default = 5
    )
    ap.add_argument(
        '--use_s3', 
        action = 'store_true'
    )
    ap.add_argument(
        '--output_path', 
        default = 'data/retrieved_chunks.jsonl'
    )
    args = ap.parse_args()

    # save_runlog(args, sub_dir = 'evaluate')

    with open(os.path.join(args.index_path, 'embedder.json')) as f:
        embedder_cfg = json.load(f)
    
    embedder = load_embedder(
        embedder_provider = embedder_cfg['embedder_provider'], 
        path = args.index_path
    )
    index = load_index(args.index_path)

    queries = load_jsonl(args.queries_path, use_s3 = args.use_s3)
    results = run_search(queries, index, embedder, k = 5)
    save_jsonl(results, args.output_path, use_s3 = args.use_s3)

if __name__ == '__main__': 
    main()


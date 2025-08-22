import argparse 

from src.embeddings.hf import HFEmbeddings
from src.embeddings.bedrock import BedrockEmbeddings

from src.vectorstore.faiss import FaissVectorStore
from src.vectorstore.pipeline import create_index

import os 
from datetime import datetime
import json 

from src.utils.runlog import save_runlog
from src.utils.io import load_jsonl

if __name__ == '__main__': 
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '--input_path', 
        default = 'data/processed/processed_chunks.json1'
    )
    ap.add_argument(
        '--use_s3', 
        action = 'store_true'
    )
    ap.add_argument(
        '--embedder_provider', 
        choices = ['hf', 'bedrock'],
        default = 'hf'
    )
    ap.add_argument(
        '--embedder_variant', 
        default = 'sentence-transformers/all-MiniLM-L6-v2'
    )
    ap.add_argument(
        'embedding_dim', 
        default = None, 
        type = int
    )
    ap.add_argument(
        '--device', 
        default = 'cpu'
    )
    ap.add_argument(
        '--metric', 
        choices = ['ip', 'l2'],
        default = 'ip'
    )
    ap.add_argument(
        '--normalize', 
        action = 'store_true'
    )
    ap.add_argument(
        '--use_ivf', 
        action = 'store_true'
    )
    ap.add_argument(
        '--nlist', 
        type = int, 
        default = 1024
    )
    ap.add_argument(
        '--output_path', 
        default = 'vector_store/faiss_store'
    )
    ap.add_argument(
        '--batch_size', 
        type = int, 
        default = 32
    )
    args = ap.parse_args()

    save_runlog(args)

    if args.embedder_provider == 'hf': 
        from src.embeddings.hf import HFEmbeddings
        embedder = HFEmbeddings(
            model_name = args.embedder_variant, 
            device = args.device if args.device != 'cpu' else None
        )
    else: 
        from src.embeddings.bedrock import BedrockEmbeddings
        embedder = BedrockEmbeddings(
            model_id = args.embedder_variant, 
            dim = args.embedding_dim, 
            client = None
        )

    if args.use_ivf: 
        vector_store = FaissVectorStore.ivf(
            dim = embedder.dim, 
            nlist = args.nlist, 
            metric = args.metric, 
            normalize = args.normalize
        )
    else: 
        vector_store = FaissVectorStore(
            dim = embedder.dim, 
            metric = args.metric, 
            normalize = args.normalize
        )

    create_index(
        records = load_jsonl(args.input_path, from_s3 = args.use_s3),
        embedder = embedder, 
        store = vector_store, 
        batch_size = args.batch_size
    )

    store.save(args.output_path)

    print(f'Vector store saved to {args.output_path}')


import argparse 
from src.preprocess.pipeline import chunk_articles
from src.utils.runlog import save_runlog
import os 
from datetime import datetime 
import json 

def main(): 
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '--provider', 
        choices = ['hf', 'bedrock'],
        default = 'hf'
    ) 
    ap.add_argument(
        '--variant',
        default = None
    )
    ap.add_argument(
        '--hf_model',
        default = 'sentence-transformers/all-MiniLM-L6-v2'
    )
    ap.add_argument(
        '--max_tokens', 
        type = int, 
        default = 400
    )
    ap.add_argument(
        '--overlap', 
        type = int, 
        default = 50
    )
    ap.add_argument(
        '--min_tokens', 
        type = int, 
        default = 50
    )
    ap.add_argument(
        '--text_key', 
        default = 'body'
    )
    ap.add_argument(
        '--id_key', 
        default = 'pmcid'
    )
    ap.add_argument(
        '--meta_keys',
        nargs = '+',             
        type = str,
        default = ['title', 'pub_date', 'doi']
    )
    ap.add_argument(
        '--input_path', 
        default = 'data/interim/parsed_xml.jsonl'
    )
    ap.add_argument(
        '--output_path', 
        default = 'data/processed/processed_chunks.jsonl'
    )
    ap.add_argument(
        '--use_s3', 
        action = 'store_true'
    )
    
    args = ap.parse_args()

    save_runlog(args, sub_dir = 'run_chunking')
    
    chunk_articles(
        provider = args.provider, 
        variant = args.variant, 
        hf_model = args.hf_model, 
        max_tokens = args.max_tokens, 
        overlap = args.overlap, 
        min_tokens = args.min_tokens, 
        text_key = args.text_key, 
        id_key = args.id_key, 
        meta_keys = args.meta_keys, 
        input_path = args.input_path, 
        output_path = args.output_path, 
        use_s3 = args.use_s3
    )

if __name__ == '__main__':
    main()
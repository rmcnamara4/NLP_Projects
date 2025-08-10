import argparse 
from src.preprocess.pipeline import chunk_articles
from src.utils.runlog import * 
import os 
from datetime import datetime 
import json 

if __name__ == '__main__': 
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

    run_id = datetime.utcnow().strftime('%Y%m%d_%H%M%S') 
    payload = {
        'run_id': run_id,
        'ts_utc': datetime.utcnow().isoformat(),
        'args': vars(args),
        'env': {
            'region': os.getenv('AWS_REGION'),
            'user': os.getenv('SAGEMAKER_USER_PROFILE_NAME'),
            'job_name': os.getenv('TRAINING_JOB_NAME') or os.getenv('PROCESSING_JOB_NAME'),
        },
    }

    # 1) Local run log (to a writable dir)
    local_dir = writable_runlog_dir()
    local_path = os.path.join(local_dir, f'run_chunking/run_{run_id}.json')
    os.makedirs(os.path.dirname(local_path), exist_ok = True)
    with open(local_path, 'w') as f:
        json.dump(payload, f, indent = 2)
    print(f'[runlog] local -> {local_path}')

    # 2) S3 run log (authoritative)
    s3_uri = s3_runlog(payload, prefix = 'run_logs/run_chunking')
    print(f'[runlog] s3 -> {s3_uri}')
    
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
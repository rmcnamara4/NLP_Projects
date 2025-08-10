import argparse
from src.data.pipeline import collect_data
from src.utils.runlog import *
import os
from datetime import datetime
import json

if __name__ == '__main__': 
    ap = argparse.ArgumentParser()
    ap.add_argument('--query', required = True)
    ap.add_argument('--max_results', type = int, default = 50)
    ap.add_argument('--date_from', default = None)
    ap.add_argument('--date_to', default = None)
    ap.add_argument('--use_s3', action = 'store_true')
    ap.add_argument(
        '--raw_key', 
        default = 'data/raw/raw_xml.jsonl'
    )
    ap.add_argument(
        '--interim_key'
        default = 'data/interim/parsed_xml.jsonl'
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
    local_path = os.path.join(local_dir, f'fetch_pmc/run_{run_id}.json')
    os.makedirs(os.path.dirname(local_path), exist_ok = True)
    with open(local_path, 'w') as f:
        json.dump(payload, f, indent = 2)
    print(f'[runlog] local -> {local_path}')

    # 2) S3 run log (authoritative)
    s3_uri = s3_runlog(payload, prefix = 'run_logs/fetch_pmc')
    print(f'[runlog] s3 -> {s3_uri}')

    collect_data(
        query = args.query, 
        max_results = args.max_results, 
        date_from = args.date_from, 
        date_to = args.date_to, 
        use_s3 = args.use_s3, 
        raw_key = args.raw_key, 
        interim_key = args.interim_key
    )



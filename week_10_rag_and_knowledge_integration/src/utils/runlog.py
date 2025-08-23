import os 
import boto3
import json
from datetime import datetime
import glob
from dotenv import load_dotenv
load_dotenv()

S3_BUCKET = os.getenv('S3_BUCKET')
s3_client = boto3.client(
    's3',
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name = os.getenv('AWS_DEFAULT_REGION', 'us-east-2')
)

def in_sm_job(): 
    return bool(
        os.environ.get('SM_CURRENT_HOST')
        or os.environ.get('TRAINING_JOB_NAME')
        or os.environ.get('PROCESSING_JOB_NAME')
    )

def writable_runlog_dir():
    if in_sm_job():
        for p in ['/opt/ml/processing/output', '/opt/ml/output', '/opt/ml/model']:
            if os.path.isdir(p):
                return os.path.join(p, 'run_logs')
    return 'run_logs'

def s3_runlog(payload, prefix = 'runs'):
    run_id = payload['run_id']
    key = f'{prefix}/{run_id}/args.json'
    s3_client.put_object(Bucket = S3_BUCKET, Key = key, Body = json.dumps(payload, indent = 2).encode('utf-8'))
    return f's3://{S3_BUCKET}/{key}'

def save_runlog(args, sub_dir = 'run'): 
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
    local_path = os.path.join(local_dir, f'{sub_dir}/run_{run_id}.json')
    os.makedirs(os.path.dirname(local_path), exist_ok = True)
    with open(local_path, 'w') as f:
        json.dump(payload, f, indent = 2)
    print(f'[runlog] local -> {local_path}')

    # 2) S3 run log (authoritative)
    s3_uri = s3_runlog(payload, prefix = f'run_logs/{sub_dir}')
    print(f'[runlog] s3 -> {s3_uri}')

def load_latest_run(log_dir: str): 
    files = sorted(glob.glob(os.path.join(log_dir, 'run_*.json')))
    if not files: 
        raise FileNotFoundError(f'No run log files found in {log_dir}')
    with open(files[-1]) as f: 
        log = json.load(f) 

    args = log['args']
    return args 


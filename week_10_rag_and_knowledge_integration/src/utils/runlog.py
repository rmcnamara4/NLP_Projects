import os 
import boto3
import json
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
    s3_client.put_object(Bucket = S3_BUCKET, Key = key, Body = json.dumps(payload).encode('utf-8'))
    return f's3://{S3_BUCKET}/{key}'
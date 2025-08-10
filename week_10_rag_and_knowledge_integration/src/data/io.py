import json 
from pathlib import Path 
from typing import Iterable, Dict, Optional
import boto3 
import os

from dotenv import load_dotenv
load_dotenv()

S3_BUCKET = os.getenv('S3_BUCKET')
s3_client = boto3.client('s3')

def save_jsonl(records: Iterable[Dict], filename: str, use_s3: Optional[bool] = True): 
    if use_s3: 
        if not S3_BUCKET: 
            raise ValueError('S3_BUCKET env variable is not set.') 
        s3_client.put_object(
            Bucket = S3_BUCKET, 
            Key = filename, 
            Body = '\n'.join(json.dumps(r) for r in records), 
            ContentType = 'application/json'
        )
        print(f'Saved to s3://{S3_BUCKET}/{filename}')
    else: 
        os.makedirs(os.path.dirname(filename), exist_ok = True) 
        with open(filename, 'w', encoding = 'utf-8') as f: 
            for r in records: 
                f.write(json.dumps(r) + '\n') 
            print(f'Saved locally to {filename}')

def load_jsonl(path: str, from_s3: Optional[bool] = True):
    if from_s3:
        if not S3_BUCKET:
            raise ValueError('S3_BUCKET env variable is not set.')
        resp = s3_client.get_object(Bucket = S3_BUCKET, Key = path)
        lines = resp['Body'].read().decode('utf-8').splitlines()
        return [json.loads(l) for l in lines if l.strip()]
    else:
        with open(path, 'r', encoding = 'utf-8') as f:
            return [json.loads(l) for l in f if l.strip()]

        

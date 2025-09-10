import json 
from pathlib import Path 
from typing import Iterable, Dict, Optional
import boto3 
import os

from dotenv import load_dotenv
load_dotenv()

S3_BUCKET = os.getenv('S3_BUCKET')
s3_client = boto3.client(
    's3',
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name = os.getenv('AWS_DEFAULT_REGION', 'us-east-2')
)

def save_jsonl(records: Iterable[Dict], filename: str, use_s3: Optional[bool] = True): 
    """
    Save a list of records in JSON Lines (JSONL) format either locally or to S3.

    Args:
        records: Iterable of dictionaries to save.
        filename: Path (local file or S3 key) where the JSONL should be stored.
        use_s3: If True (default), save to S3. If False, save to local filesystem.

    Raises:
        ValueError: If `use_s3=True` but the `S3_BUCKET` environment variable is not set.

    Returns:
        None. Saves the file locally or to the specified S3 bucket and prints the save location.
    """
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

def load_jsonl(path: str, use_s3: Optional[bool] = True):
    """
    Load records from a JSON Lines (JSONL) file stored locally or in S3.

    Args:
        path: File path (for local) or S3 key (for remote) to read from.
        use_s3: If True (default), read from S3. If False, read from local filesystem.

    Raises:
        ValueError: If `use_s3=True` but the `S3_BUCKET` environment variable is not set.

    Returns:
        List[dict]: A list of dictionaries, one per JSONL line.
    """
    if use_s3:
        if not S3_BUCKET:
            raise ValueError('S3_BUCKET env variable is not set.')
        resp = s3_client.get_object(Bucket = S3_BUCKET, Key = path)
        lines = resp['Body'].read().decode('utf-8').splitlines()
        return [json.loads(l) for l in lines if l.strip()]
    else:
        with open(path, 'r', encoding = 'utf-8') as f:
            return [json.loads(l) for l in f if l.strip()]

        

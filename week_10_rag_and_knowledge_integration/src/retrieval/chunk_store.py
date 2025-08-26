import json 
from functools import lru_cache 
from typing import Dict, Iterable 

from src.utils.io import load_jsonl

@lru_cache(maxsize = 1)
def _load_all_chunks(path: str, use_s3: bool = True) -> Dict[str, Dict]: 
    chunks = load_jsonl(path, use_s3 = use_s3)
    chunk_dict = {chunk['id']: chunk for chunk in chunks}
    return chunk_dict

def get_chunk_text(chunk_id: str, path: str, use_s3: bool = True) -> str: 
    chunks = _load_all_chunks(path, use_s3 = use_s3)
    chunk = chunks.get(chunk_id, None)
    if chunk is None: 
        raise KeyError(f'Chunk id {chunk_id} not found in chunk store.')
    return chunk['text']

def batch_get_texts(chunk_ids: Iterable[str], path: str, use_s3: bool = True) -> Dict[str, str]: 
    chunks = _load_all_chunks(path, use_s3 = use_s3)
    return {cid: chunks[cid]['text'].strip() for cid in chunk_ids if cid in chunks}
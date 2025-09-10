import json 
from functools import lru_cache 
from typing import Dict, Iterable 

from src.utils.io import load_jsonl

@lru_cache(maxsize = 1)
def _load_all_chunks(path: str, use_s3: bool = True) -> Dict[str, Dict]: 
    """
    Load all text chunks from a JSONL file or S3 into a dictionary.

    Args:
        path: Path to the JSONL file containing chunk records.
        use_s3: Whether to load from S3 storage. Defaults to True.

    Returns:
        Dict[str, Dict]: Mapping of chunk IDs to their full record metadata.
    """
    chunks = load_jsonl(path, use_s3 = use_s3)
    chunk_dict = {chunk['id']: chunk for chunk in chunks}
    return chunk_dict

def get_chunk_text(chunk_id: str, path: str, use_s3: bool = True) -> str: 
    """
    Retrieve the text content of a specific chunk by its ID.

    Args:
        chunk_id: Unique identifier of the chunk.
        path: Path to the JSONL file containing chunk records.
        use_s3: Whether to load from S3 storage. Defaults to True.

    Returns:
        str: The text content of the specified chunk.

    Raises:
        KeyError: If the given chunk_id does not exist in the chunk store.
    """
    chunks = _load_all_chunks(path, use_s3 = use_s3)
    chunk = chunks.get(chunk_id, None)
    if chunk is None: 
        raise KeyError(f'Chunk id {chunk_id} not found in chunk store.')
    return chunk['text']

def batch_get_texts(chunk_ids: Iterable[str], path: str, use_s3: bool = True) -> Dict[str, str]: 
    """
    Retrieve text content for multiple chunks by their IDs.

    Args:
        chunk_ids: List or iterable of chunk IDs to fetch.
        path: Path to the JSONL file containing chunk records.
        use_s3: Whether to load from S3 storage. Defaults to True.

    Returns:
        Dict[str, str]: Mapping of chunk IDs to their corresponding text content.
    """
    chunks = _load_all_chunks(path, use_s3 = use_s3)
    return {cid: chunks[cid]['text'].strip() for cid in chunk_ids if cid in chunks}
from src.preprocess.chunking import chunk_many, get_tokenizer
from src.utils.io import *
from typing import List

def chunk_articles(
        provider: str, 
        variant: str = None, 
        hf_model: str = None, 
        max_tokens: int = 400, 
        overlap: int = 50, 
        min_tokens: int = 50, 
        text_key: str = 'body', 
        id_key: str = 'pmcid', 
        meta_keys: List[str] = ['title', 'pub_date', 'doi'], 
        input_path: str = 'data/interim/parsed_xml.jsonl', 
        output_path: str = 'data/processed/processed_chunks.jsonl', 
        use_s3: bool = True
): 
    """
    Load articles from JSONL, tokenize, split into chunks, and save processed output.

    Uses the specified tokenizer (Hugging Face or Bedrock-based) to break large
    text fields into overlapping chunks suitable for embedding and retrieval tasks.
    Metadata is preserved with each chunk.

    Args:
        provider: Provider of tokenizer ('hf' for Hugging Face, 'bedrock' for Bedrock).
        variant: Variant string to select a specific Bedrock tokenizer family.
        hf_model: Hugging Face model name to use if provider='hf'.
        max_tokens: Maximum tokens per chunk. Defaults to 400.
        overlap: Number of overlapping tokens between adjacent chunks. Defaults to 50.
        min_tokens: Minimum number of tokens required to keep a chunk. Defaults to 50.
        text_key: Key in the JSON record containing text to chunk. Defaults to 'body'.
        id_key: Key in the JSON record for unique IDs. Defaults to 'pmcid'.
        meta_keys: Keys to include as metadata in each chunk. Defaults to ['title', 'pub_date', 'doi'].
        input_path: Path to input JSONL file with article records. Defaults to 'data/interim/parsed_xml.jsonl'.
        output_path: Path to save processed chunks as JSONL. Defaults to 'data/processed/processed_chunks.jsonl'.
        use_s3: Whether to load/save from S3 storage. Defaults to True.

    Returns:
        None: Writes chunked article records to the specified output path.
    """
    records = load_jsonl(input_path, use_s3 = use_s3)

    tokenizer = get_tokenizer(
        provider, 
        variant, 
        hf_model
    )

    chunked_records = chunk_many(
        records, 
        tokenizer, 
        text_key, 
        id_key, 
        meta_keys, 
        max_tokens, 
        overlap, 
        min_tokens
    )

    save_jsonl(chunked_records, output_path, use_s3 = use_s3)


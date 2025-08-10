from src.preprocess.chunking import chunk_many, get_tokenizer
from src.utils.io import *

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


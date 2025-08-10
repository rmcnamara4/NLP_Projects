from transformers import AutoTokenizer
from typing import Dict, List, Iterable
from src.preprocess.normalize import clean_text

def get_tokenizer(provider: str, variant: str = None, hf_model: str = None): 
    provider = (provider or '').lower()
    variant = (variant or '').lower() if variant else None

    if provider == 'bedrock' and variant in ('titan', 'amazon.titan-embed'): 
        return AutoTokenizer.from_pretrained('bert-base-uncased') 
    
    if provider == 'bedrock' and variant in ('cohere-en', 'cohere-embed-en'): 
        return AutoTokenizer.from_pretrained('gpt2') 
    
    if provider in ('hf', 'huggingface') and hf_model: 
        return AutoTokenizer.from_pretrained(hf_model) 
    
    return None

def chunk_text(text: str, tokenizer, max_tokens: int = 400, overlap: int = 50, min_tokens: int = 50) -> List[str]: 
    text = clean_text(text)
    ids = tokenizer.encode(text, add_special_tokens = False) 
    chunks = []
    start = 0
    n = len(ids) 

    while start < n: 
        end = min(start + max_tokens, n)
        piece_ids = ids[start:end]

        if len(piece_ids) >= min_tokens: 
            chunk_text = tokenizer.decode(piece_ids, skip_special_tokens = True) 
            chunks.append(chunk_text) 

        if end == n: 
            break

        start = end - overlap if end - overlap > start else end 

    return chunks  

def chunk_record(
    record: Dict, 
    tokenizer, 
    text_key: str = 'body', 
    id_key: str = 'pmcid', 
    meta_keys: List[str] = ['title'],
    max_tokens: int = 400, 
    overlap: int = 50, 
    min_tokens: int = 50
) -> List[Dict]: 
    text = record.get(text_key, '') or ''
    metadata = {m: record.get(m, None) for m in meta_keys}
    id = record.get(id_key, '') or None

    parts = chunk_text(text, tokenizer, max_tokens, overlap, min_tokens) 
    out = []
    for i, part in enumerate(parts): 
        item = {
            'id': f'{id}::chunk_{i}' if id else f'chunk_{i}', 
            'chunk_index': i, 
            'text': part, 
            **metadata
        }
        out.append(item)

    return out 

def chunk_many(
    records: Iterable[Dict], 
    tokenizer, 
    text_key: str = 'body', 
    id_key: str = 'pmcid', 
    meta_keys: List[str] = ['title'], 
    max_tokens: int = 400, 
    overlap: int = 50, 
    min_tokens: int = 50
) -> List[Dict]: 
    output = []
    for r in records: 
        output.extend(chunk_record(r, tokenizer, text_key, id_key, meta_keys, max_tokens, overlap, min_tokens))
    return output



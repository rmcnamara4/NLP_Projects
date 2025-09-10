from transformers import AutoTokenizer
from typing import Dict, List, Iterable
from src.preprocess.normalize import clean_text

def get_tokenizer(provider: str, variant: str = None, hf_model: str = None): 
    """
    Return a Hugging Face tokenizer based on provider and model variant.

    Args:
        provider (str): Provider name (e.g., "bedrock", "hf", "huggingface").
        variant (str, optional): Model variant identifier for Bedrock embeddings
            (e.g., "titan", "amazon.titan-embed", "cohere-en", "cohere-embed-en").
        hf_model (str, optional): Hugging Face model name to load if provider is "hf" or "huggingface".

    Returns:
        AutoTokenizer | None: A Hugging Face tokenizer instance if a valid configuration
        is provided, otherwise None.

    Examples:
        >>> tok = get_tokenizer("bedrock", variant="titan")
        >>> tok("Sample text")  # Tokenize with bert-base-uncased

        >>> tok = get_tokenizer("hf", hf_model="sentence-transformers/all-MiniLM-L6-v2")
        >>> tok("Another example")
    """
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
    """
    Splits a text into overlapping chunks based on token counts.

    Args:
        text (str): Raw input text to be chunked.
        tokenizer: Tokenizer with `encode` and `decode` methods (e.g., Hugging Face tokenizer).
        max_tokens (int, optional): Maximum number of tokens per chunk. Defaults to 400.
        overlap (int, optional): Number of tokens to overlap between consecutive chunks. Defaults to 50.
        min_tokens (int, optional): Minimum tokens required to include a chunk. Defaults to 50.

    Returns:
        List[str]: List of decoded text chunks that meet the size constraints.
    """
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
    """
    Splits the text of a single record into token-based chunks and attaches metadata.

    Args:
        record (Dict): Input record containing text and metadata fields. 
        tokenizer: Tokenizer with `encode`/`decode` methods for splitting text into tokens. 
        text_key (str, optional): Key in `record` holding the main text. Defaults to 'body'.
        id_key (str, optional): Key in `record` used to build chunk IDs. Defaults to 'pmcid'.
        meta_keys (List[str], optional): Keys in `record` to carry over as metadata. Defaults to ['title'].
        max_tokens (int, optional): Maximum tokens per chunk. Defaults to 400.
        overlap (int, optional): Number of overlapping tokens between chunks. Defaults to 50.
        min_tokens (int, optional): Minimum tokens required for a valid chunk. Defaults to 50.

    Returns:
        List[Dict]: A list of chunk dictionaries, each with:
            - 'id': Unique chunk identifier (combining record ID + chunk index).
            - 'chunk_index': Position of the chunk within the record.
            - 'text': Chunked text content.
            - Additional metadata fields copied from `meta_keys`.
    """
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
    """
    Chunk multiple records into smaller text segments with metadata.

    Iterates over a collection of records and applies `chunk_record` to each,
    producing token-length–bounded chunks enriched with record identifiers and metadata.

    Args:
        records: Iterable of dictionaries, each representing a record with text and metadata.
        tokenizer: Tokenizer object with `encode` and `decode` methods (e.g., Hugging Face tokenizer).
        text_key: Key in each record that contains the text to be chunked. Defaults to "body".
        id_key: Key used to assign a unique ID to each record. Defaults to "pmcid".
        meta_keys: Keys from each record to include in the output metadata. Defaults to ["title"].
        max_tokens: Maximum number of tokens per chunk. Defaults to 400.
        overlap: Number of tokens to overlap between consecutive chunks. Defaults to 50.
        min_tokens: Minimum number of tokens required for a chunk to be kept. Defaults to 50.

    Returns:
        List[Dict]: List of chunked records, each containing an ID, chunk index,
        text content, and any included metadata fields.
    """
    output = []
    for r in records: 
        output.extend(chunk_record(r, tokenizer, text_key, id_key, meta_keys, max_tokens, overlap, min_tokens))
    return output



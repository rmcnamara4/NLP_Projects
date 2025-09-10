import json 
from pathlib import Path 
from typing import Dict, Any, Iterable, List, Tuple 
from functools import lru_cache

from src.llm.prompting import build_prompt, build_system_prompt
from src.llm.bedrock_llm import BedrockLLM
from src.utils.io import load_jsonl

@lru_cache(maxsize = 1) 
def _load_retrieved_json(path: str, use_s3: bool = True) -> Dict[str, Any]: 
    """
    Load and cache retrieved results from a JSONL file.

    Args:
        path (str): Path to the JSONL file containing retrieval results.
        use_s3 (bool, optional): If True, load the file from S3. Defaults to True.

    Returns:
        Dict[str, Any]: A mapping from chunk/document IDs (`id`) to the
        corresponding metadata dictionaries.
    """
    retrieved_ids = load_jsonl(path, use_s3) 
    return {
        m['id']: m
        for m in retrieved_ids
    }

def batch_get_texts(chunk_ids: Iterable[str], path: str, use_s3: bool = True) -> Dict[str, str]: 
    """
    Retrieve and return text content for a batch of chunk IDs.

    Args:
        chunk_ids (Iterable[str]): List or iterable of chunk/document IDs to fetch.
        path (str): Path to the JSONL file containing retrieval results.
        use_s3 (bool, optional): If True, load the file from S3. Defaults to True.

    Returns:
        Dict[str, str]: Mapping from chunk IDs to their corresponding text
        (stripped of leading/trailing whitespace).
    """
    retrieved = _load_retrieved_json(path, use_s3) 
    out = {}
    for cid in chunk_ids: 
        rec = retrieved[cid]
        if rec and 'text' in rec: 
            out[cid] = rec['text'].strip()
    return out 

def _format_block(i: int, rec: Dict[str, Any], text: str) -> str: 
    """
    Format a text block with metadata for inclusion as context in prompt.

    Args:
        i (int): Index or reference number for the block.
        rec (Dict[str, Any]): Metadata dictionary containing keys such as
            'title', 'pub_date', and 'doi'.
        text (str): The main text content to include in the block.

    Returns:
        str: A formatted string including the index, title, publication date,
        DOI (if available), and the provided text content.
    """
    title = rec.get('title') or 'Unknown Title'
    pub_date = rec.get('pub_date') or 'n.d.'
    doi = rec.get('doi') or 'n.d.'
    return (
        f'[{i}] "{title}" (pub: {pub_date}; doi: {doi})\n'
        f'{text}\n'
    )

def build_context_window(
    retrieved: List[Dict[str, Any]], 
    chunks_json_path: str, 
    use_s3: bool = True, 
    max_chars: int = 6000
) -> Tuple[List[str], List[Dict[str, Any]]]: 
    """
    Build a context window of retrieved chunks with metadata, constrained by a character budget.

    Args:
        retrieved (List[Dict[str, Any]]): List of retrieved chunk metadata dictionaries,
            each containing at least an 'id' field.
        chunks_json_path (str): Path to the JSONL file containing chunk texts and metadata.
        use_s3 (bool, optional): If True, load data from S3. Defaults to True.
        max_chars (int, optional): Maximum total number of characters allowed in the
            concatenated context. Defaults to 6000.

    Returns:
        Tuple[List[str], List[Dict[str, Any]]]:
            - contexts (List[str]): Formatted text blocks including title, pub date, DOI,
              and content, truncated to the character budget.
            - used_meta (List[Dict[str, Any]]): Metadata dictionaries for the chunks
              actually included in the context window.
    """
    seen, ordered = set(), []
    for r in retrieved: 
        cid = r['id']
        if cid not in seen: 
            seen.add(cid) 
            ordered.append(r) 

    ids = [r['id'] for r in ordered]
    texts = batch_get_texts(ids, chunks_json_path, use_s3) 
    
    contexts, used_meta = [], []
    char_budget = 0
    i = 1
    for r in ordered: 
        cid = r['id']

        if cid not in texts: 
            continue 

        rec_text = texts[cid]
        block = _format_block(i, r, rec_text) 

        if char_budget + len(block) > max_chars: 
            break 

        contexts.append(block) 
        used_meta.append(r) 

        char_budget += len(block) 
        i += 1

    return contexts, used_meta

def answer_with_rag(
    question: str, 
    retrieved_results: List[Dict[str, Any]], 
    chunks_json_path: str, 
    use_s3: bool, 
    max_chars: int, 
    prompt_dir: str,
    llm: BedrockLLM, 
    **llm_overrides
) -> Dict[str, Any]: 
    """
    Generate an answer to a question using Retrieval-Augmented Generation (RAG).

    This function builds a context window from retrieved results, constructs prompts,
    and queries the provided LLM to produce an answer grounded in retrieved evidence.  

    Args:
        question (str): The input question to answer.
        retrieved_results (List[Dict[str, Any]]): Retrieved chunk metadata, each with at least an 'id'.
        chunks_json_path (str): Path to the JSONL file containing chunk texts and metadata.
        use_s3 (bool): Whether to load chunks from S3.
        max_chars (int): Maximum number of characters allowed for the combined context window.
        prompt_dir (str): Directory containing system and user prompt templates.
        llm (BedrockLLM): LLM instance used to generate the answer.
        **llm_overrides: Optional overrides for LLM generation parameters
            (e.g., temperature, max_tokens).

    Returns:
        Dict[str, Any]: A dictionary containing:
            - "question" (str): The original question.
            - "answer" (str): The LLM-generated answer.
            - "used_contexts" (List[Dict[str, Any]]): Metadata for contexts included in the prompt.
            - "raw" (Any): Raw LLM response payload for inspection/debugging.
    """
    contexts, used_meta = build_context_window(retrieved_results, chunks_json_path, use_s3, max_chars)
    user_prompt = build_prompt(prompt_dir, question, contexts)
    system_prompt = build_system_prompt(prompt_dir)

    messages = [
        {'role': 'system', 'content': system_prompt}, 
        {'role': 'user', 'content': user_prompt}
    ]

    resp = llm.generate(messages, **llm_overrides)

    return {
        'question': question, 
        'answer': resp.get('text', ''), 
        'used_contexts': used_meta, 
        'raw': resp.get('raw')
    }




import json 
from pathlib import Path 
from typing import Dict, Any, Iterable, List, Tuple 
from functools import lru_cache

from src.llm.prompting import build_prompt, build_system_prompt
from src.llm.bedrock_llm import BedrockLLM
from src.utils.io import load_jsonl

@lru_cache(maxsize = 1) 
def _load_retrieved_json(path: str, use_s3: bool = True) -> Dict[str, Any]: 
    retrieved_ids = load_jsonl(path, use_s3) 
    return {
        m['id']: m
        for m in retrieved_ids
    }

def batch_get_texts(chunk_ids: Iterable[str], path: str, use_s3: bool = True) -> Dict[str, str]: 
    retrieved = _load_retrieved_json(path, use_s3) 
    out = {}
    for cid in chunk_ids: 
        rec = retrieved[cid]
        if rec and 'text' in rec: 
            out[cid] = rec['text'].strip()
    return out 

def _format_block(i: int, rec: Dict[str, Any], text: str) -> str: 
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




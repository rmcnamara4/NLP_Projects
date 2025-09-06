from typing import List, Callable, Dict, Any
import numpy as np 
from scipy.spatial.distance import cosine 

# -------------------------------------- matchers --------------------------------------
def id_match(retrieved_id: str, golden_ids: set) -> bool: 
    return retrieved_id in golden_ids

def semantic_match(retrieved_vec, golden_vecs, sim_threshold = 0.8) -> bool: 
    sims = [1 - cosine(retrieved_vec, g) for g in golden_vecs]
    return max(sims) >= sim_threshold

# -------------------------------------- metrics --------------------------------------
def hit_at_k(retrieved: List[Any], golden: set, k: int, is_match: Callable) -> float: 
    return float(any(is_match(r, golden) for r in retrieved[:k]))

def recall_at_k(retrieved: List[Any], golden: set, k: int, is_match: Callable) -> float: 
    hits = [r for r in retrieved[:k] if is_match(r, golden)]
    return len(hits) / len(golden) if golden else 0.0

def precision_at_k(retrieved: List[Any], golden: set, k: int, is_match: Callable) -> float: 
    hits = sum(is_match(r, golden) for r in retrieved[:k])
    return hits / max(k, 1)

def mrr(retrieved: List[Any], golden: set, is_match: Callable) -> float: 
    for i, r in enumerate(retrieved, start = 1): 
        if is_match(r, golden): 
            return 1.0 / i
    return 0.0 

def coverage(all_retrieved: List[List[Any]], all_golden: List[set], is_match: Callable) -> float: 
    use_union = True 
    try: 
        gold_union = set().union(*all_golden)
    except TypeError: 
        use_union = False 

    if use_union: 
        seen = set()
        for recs in all_retrieved: 
            for r in recs: 
                if is_match(r, gold_union): 
                    seen.add(r) 
        return len(seen) / max(len(gold_union), 1)
    
    covered_flags = []
    for recs, gold in zip(all_retrieved, all_golden): 
        if not gold: 
            continue 
        covered_flags.append(any(is_match(r, gold) for r in recs))
    return (sum(covered_flags) / len(covered_flags)) if covered_flags else 0.0


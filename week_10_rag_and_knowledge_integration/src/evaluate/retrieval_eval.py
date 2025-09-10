from typing import List, Callable, Dict, Any
import numpy as np 
from scipy.spatial.distance import cosine 

# -------------------------------------- matchers --------------------------------------
def id_match(retrieved_id: str, golden_ids: set) -> bool: 
    """
    Check if a retrieved identifier matches one of the golden identifiers.

    Args:
        retrieved_id (str): The identifier returned by the retrieval system.
        golden_ids (set): Set of ground-truth identifiers considered correct.

    Returns:
        bool: True if the retrieved_id is in golden_ids, otherwise False.
    """
    return retrieved_id in golden_ids

def semantic_match(retrieved_vec, golden_vecs, sim_threshold = 0.8) -> bool: 
    """
    Check if a retrieved vector semantically matches any golden vectors 
    based on cosine similarity.

    Args:
        retrieved_vec (np.ndarray): Embedding vector for the retrieved item.
        golden_vecs (List[np.ndarray]): List of embedding vectors representing
            ground-truth items.
        sim_threshold (float, optional): Cosine similarity threshold for a match.
            Defaults to 0.8.

    Returns:
        bool: True if the retrieved vector has cosine similarity >= sim_threshold
        with any golden vector, otherwise False.
    """
    sims = [1 - cosine(retrieved_vec, g) for g in golden_vecs]
    return max(sims) >= sim_threshold

# -------------------------------------- metrics --------------------------------------
def hit_at_k(retrieved: List[Any], golden: set, k: int, is_match: Callable) -> float: 
    """
    Check if at least one of the top-k retrieved items matches any golden item.

    Args:
        retrieved (List[Any]): Ranked list of retrieved items.
        golden (set): Ground-truth items.
        k (int): Cutoff rank to consider.
        is_match (Callable): Function to determine whether an item matches.

    Returns:
        float: 1.0 if any match occurs in top-k, otherwise 0.0.
    """
    return float(any(is_match(r, golden) for r in retrieved[:k]))

def recall_at_k(retrieved: List[Any], golden: set, k: int, is_match: Callable) -> float: 
    """
    Compute recall at k, i.e. fraction of golden items found in top-k retrieved.

    Args:
        retrieved (List[Any]): Ranked list of retrieved items.
        golden (set): Ground-truth items.
        k (int): Cutoff rank to consider.
        is_match (Callable): Function to determine whether an item matches.

    Returns:
        float: Recall value in [0, 1]. Returns 0.0 if no golden items exist.
    """
    hits = [r for r in retrieved[:k] if is_match(r, golden)]
    return len(hits) / len(golden) if golden else 0.0

def precision_at_k(retrieved: List[Any], golden: set, k: int, is_match: Callable) -> float: 
    """
    Compute precision at k, i.e. fraction of retrieved items in top-k that are correct.

    Args:
        retrieved (List[Any]): Ranked list of retrieved items.
        golden (set): Ground-truth items.
        k (int): Cutoff rank to consider.
        is_match (Callable): Function to determine whether an item matches.

    Returns:
        float: Precision value in [0, 1].
    """
    hits = sum(is_match(r, golden) for r in retrieved[:k])
    return hits / max(k, 1)

def mrr(retrieved: List[Any], golden: set, is_match: Callable) -> float: 
    """
    Compute Mean Reciprocal Rank (MRR) for a single query.

    Args:
        retrieved (List[Any]): Ranked list of retrieved items.
        golden (set): Ground-truth items.
        is_match (Callable): Function to determine whether an item matches.

    Returns:
        float: Reciprocal rank of the first correct item, or 0.0 if none found.
    """
    for i, r in enumerate(retrieved, start = 1): 
        if is_match(r, golden): 
            return 1.0 / i
    return 0.0 

def coverage(all_retrieved: List[List[Any]], all_golden: List[set], is_match: Callable) -> float: 
    """
    Compute coverage, i.e. fraction of unique golden items matched across all queries.

    Args:
        all_retrieved (List[List[Any]]): Retrieved lists per query.
        all_golden (List[set]): Sets of golden items per query.
        is_match (Callable): Function to determine whether an item matches.

    Returns:
        float: Coverage value in [0, 1], measuring how much of the golden set 
        is covered across all queries.
    """
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


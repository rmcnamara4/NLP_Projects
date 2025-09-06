from src.evaluate.retrieval_eval import * 
import numpy as np 
from functools import partial

def eval_retrieval(all_retrieved: List[List[Any]], all_golden: List[set], k: int, match_type: str = 'binary', sim_threshold: float = 0.8): 
    if match_type == 'binary': 
        matcher = id_match
    elif match_type == 'semantic': 
        matcher = partial(semantic_match, sim_threshold = sim_threshold)
    else: 
        raise ValueError('match_type not one of the allowed values. Choose either "binary" or "semantic".')
    
    hit_at_k_results, recall_at_k_results, precision_at_k_results, mrr_results = [], [], [], []
    for retrieved, golden in zip(all_retrieved, all_golden): 
        if golden: 
            hit_at_k_results.append(hit_at_k(retrieved, golden, k, matcher))
            recall_at_k_results.append(recall_at_k(retrieved, golden, k, matcher))
            precision_at_k_results.append(precision_at_k(retrieved, golden, k, matcher))
            mrr_results.append(mrr(retrieved, golden, matcher))

    coverage_results = coverage(all_retrieved, all_golden, matcher) 

    return {
        f'hit_at_{k}': np.mean(hit_at_k_results), 
        f'recall_at_{k}': np.mean(recall_at_k_results), 
        f'precision_at_{k}': np.mean(precision_at_k_results), 
        'mrr': np.mean(mrr_results), 
        'coverage': coverage_results
    }
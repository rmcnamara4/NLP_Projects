from src.evaluate.retrieval_eval import * 
from src.evaluate.llm_as_judge import * 
from src.llm.pipeline import batch_get_texts
from src.utils.io import load_jsonl
import numpy as np 
from functools import partial
from typing import List, Dict, Tuple

def get_retrieved_ids(path: str, use_s3: bool = False): 
    all_retrieved = []
    retrieved_ids = load_jsonl(path, use_s3)
    for c in retrieved_ids: 
        temp = []
        for r in c['results']: 
            temp.append(r['id'])
        all_retrieved.append(temp) 
    return all_retrieved

def get_golden_ids(path: str, use_s3: bool = False) -> List[str]: 
    all_golden = []
    queries = load_jsonl(path, use_s3) 
    for q in queries: 
        all_golden.append(set(q['gold_chunks']))
    return all_golden

def get_text(ids: List[List[str]], path: str, use_s3: bool = True) -> List[List[str]]:
    texts = []
    for i in ids:  
        batch = batch_get_texts(i, path, use_s3)
        texts.append(list(batch.values()))
    return texts


def get_questions_and_answers(path: str, use_s3: bool = False) -> Tuple[List[str], List[str]]: 
    questions, answers = [], []
    results = load_jsonl(path, use_s3) 
    for r in results: 
        questions.append(r['question'])
        answers.append(r['answer'])
    return questions, answers 

def get_meta_to_int(path: str, use_s3: bool = False) -> Dict[str, int]: 
    metadata = load_jsonl(path, use_s3) 
    meta_to_int = {}
    for i, m in enumerate(metadata): 
        meta_to_int[m['id']] = i
    return meta_to_int

def get_vecs_from_ids(ids: List[str], index, meta_to_int: Dict[str, int]): 
    vecs = []
    for c in ids: 
        temp = []
        for r in c: 
            ind = meta_to_int[r]
            vec = index.reconstruct(ind) 
            temp.append(vec) 
        vecs.append(vec) 
    return vecs 

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
        else: 
            hit_at_k_results.append(np.nan) 
            recall_at_k_results.append(np.nan) 
            precision_at_k_results.append(np.nan) 
            mrr_results.append(np.nan)

    coverage_results = coverage(all_retrieved, all_golden, matcher) 

    return {
        f'hit_at_{k}': np.nanmean(hit_at_k_results), 
        f'recall_at_{k}': np.nanmean(recall_at_k_results), 
        f'precision_at_{k}': np.nanmean(precision_at_k_results), 
        'mrr': np.nanmean(mrr_results), 
        'coverage': coverage_results
    }, (hit_at_k_results, recall_at_k_results, precision_at_k_results, mrr_results)

def eval_correctness_and_faithfulness(
    llm, 
    prompt_dir: str, 
    system_template_name: str, 
    faithfulness_user_template_name: str, 
    correctness_user_template_name: str, 
    all_questions: List[str], 
    all_answers: List[str], 
    all_retrieved_texts: List[List[str]], 
    all_golden_texts: List[List[str]]
): 
    faithfulness_results = []
    correctness_results = []

    for q, a, ret_txts, gold_txts in zip(all_questions, all_answers, all_retrieved_texts, all_golden_texts): 
        f = judge_faithfulness(
            llm, 
            prompt_dir, 
            system_template_name, 
            faithfulness_user_template_name, 
            q, 
            a, 
            ret_txts
        )

        c = judge_correctness(
            llm, 
            prompt_dir, 
            system_template_name, 
            correctness_user_template_name, 
            q, 
            a, 
            gold_txts
        )

        faithfulness_results.append(f) 
        correctness_results.append(c) 

    judge_metrics = aggregate_judge_results(
        faithfulness_results, correctness_results, 
        has_gold_flags = [bool(g) for g in all_golden_texts]
    )

    return judge_metrics, (faithfulness_results, correctness_results)
from src.evaluate.retrieval_eval import * 
from src.evaluate.llm_as_judge import * 
from src.llm.pipeline import batch_get_texts
from src.utils.io import load_jsonl
import numpy as np 
from functools import partial
from typing import List, Dict, Tuple

def get_retrieved_ids(path: str, use_s3: bool = False): 
    """
    Load and extract retrieved document IDs from a JSONL results file.

    This function reads a JSONL file (either locally or from S3) containing
    retrieval results, then collects the `id` field from each result entry.

    Args:
        path (str): Path to the JSONL file. Can be a local path or an S3 key
            depending on `use_s3`.
        use_s3 (bool, optional): If True, load the file from AWS S3.
            Defaults to False.

    Returns:
        List[List[str]]: A nested list of retrieved IDs, where each inner list
        corresponds to one query and contains the IDs of its retrieved results.
    """
    all_retrieved = []
    retrieved_ids = load_jsonl(path, use_s3)
    for c in retrieved_ids: 
        temp = []
        for r in c['results']: 
            temp.append(r['id'])
        all_retrieved.append(temp) 
    return all_retrieved

def get_golden_ids(path: str, use_s3: bool = False) -> List[str]: 
    """
    Load and extract sets of golden (ground-truth) chunk IDs from a JSONL file.

    This function reads a JSONL file (either locally or from S3) containing
    queries and their associated `gold_chunks`, then converts each list of
    gold chunks into a set.

    Args:
        path (str): Path to the JSONL file. Can be a local path or an S3 key
            depending on `use_s3`.
        use_s3 (bool, optional): If True, load the file from AWS S3.
            Defaults to False.

    Returns:
        List[set]: A list of sets, where each set contains the golden chunk
        IDs for a given query.
    """
    all_golden = []
    queries = load_jsonl(path, use_s3) 
    for q in queries: 
        all_golden.append(set(q['gold_chunks']))
    return all_golden

def get_text(ids: List[List[str]], path: str, use_s3: bool = True) -> List[List[str]]:
    """
    Retrieve and organize text content for batches of IDs.

    This function looks up text values for each batch of IDs using
    `batch_get_texts`, and returns them grouped in the same nested
    structure as the input.

    Args:
        ids (List[List[str]]): Nested list of IDs, where each inner list
            corresponds to a batch of IDs whose text should be retrieved.
        path (str): Path to the JSONL file or S3 key containing ID-to-text
            mappings.
        use_s3 (bool, optional): If True, load the file from AWS S3.
            Defaults to True.

    Returns:
        List[List[str]]: A nested list of texts corresponding to the input IDs.
    """
    texts = []
    for i in ids:  
        batch = batch_get_texts(i, path, use_s3)
        texts.append(list(batch.values()))
    return texts

def get_questions_and_answers(path: str, use_s3: bool = False) -> Tuple[List[str], List[str]]: 
    """
    Load questions and answers from a JSONL file or S3.

    Each entry in the file is expected to contain "question" and "answer" keys.
    The function returns parallel lists of questions and answers.

    Args:
        path (str): Path to the JSONL file or S3 key containing data.
        use_s3 (bool, optional): If True, load the file from AWS S3.
            Defaults to False.

    Returns:
        Tuple[List[str], List[str]]: 
            - questions: A list of question strings.
            - answers: A list of corresponding answer strings.
    """
    questions, answers = [], []
    results = load_jsonl(path, use_s3) 
    for r in results: 
        questions.append(r['question'])
        answers.append(r['answer'])
    return questions, answers 

def get_meta_to_int(path: str, use_s3: bool = False) -> Dict[str, int]: 
    """
    Create a mapping from metadata IDs to their integer indices.

    Loads metadata from a JSONL file or S3 and assigns each entry
    a unique integer index based on its position in the file.

    Args:
        path (str): Path to the JSONL file or S3 key containing metadata.
        use_s3 (bool, optional): If True, load the file from AWS S3.
            Defaults to False.

    Returns:
        Dict[str, int]: A dictionary mapping each metadata "id" to
        its corresponding integer index.
    """
    metadata = load_jsonl(path, use_s3) 
    meta_to_int = {}
    for i, m in enumerate(metadata): 
        meta_to_int[m['id']] = i
    return meta_to_int

def get_vecs_from_ids(ids: List[str], index, meta_to_int: Dict[str, int]): 
    """
    Retrieve vectors from a FAISS index based on metadata IDs.

    For each metadata ID, this function uses its mapped integer index
    to reconstruct the stored vector from the FAISS index.

    Args:
        ids (List[str]): A nested list of metadata IDs grouped by query.
        index: A FAISS index object that supports vector reconstruction.
        meta_to_int (Dict[str, int]): Mapping from metadata IDs to their
            corresponding integer indices in the FAISS index.

    Returns:
        List[List[np.ndarray]]: Nested list of vectors corresponding to
        the input IDs, grouped in the same structure as the input.
    """
    vecs = []
    for c in ids: 
        temp = []
        for r in c: 
            ind = meta_to_int[r]
            vec = index.reconstruct(ind) 
            temp.append(vec) 
        vecs.append(temp) 
    return vecs 

def eval_retrieval(all_retrieved: List[List[Any]], all_golden: List[set], k: int, match_type: str = 'binary', sim_threshold: float = 0.8): 
    """
    Evaluate retrieval performance against golden sets using standard IR metrics.

    This function computes retrieval metrics such as Hit@K, Recall@K, 
    Precision@K, Mean Reciprocal Rank (MRR), and coverage. It supports 
    both exact ID matching and semantic similarity matching.

    Args:
        all_retrieved (List[List[Any]]): Nested list of retrieved results per query. 
            Each inner list contains either IDs or vectors, depending on match_type.
        all_golden (List[set]): List of golden (ground-truth) sets, one per query.
        k (int): Cutoff rank for metrics (e.g., top-k retrieved results).
        match_type (str, optional): Matching method, either "binary" (ID match) 
            or "semantic" (vector similarity). Defaults to "binary".
        sim_threshold (float, optional): Cosine similarity threshold for semantic
            matching. Defaults to 0.8.

    Returns:
        Tuple[Dict[str, float], Tuple[List[float], List[float], List[float], List[float]]]: 
            - Dict with aggregated retrieval metrics:
                - f"hit_at_{k}": Mean Hit@K across queries.
                - f"recall_at_{k}": Mean Recall@K across queries.
                - f"precision_at_{k}": Mean Precision@K across queries.
                - "mrr": Mean Reciprocal Rank.
                - "coverage": Proportion of gold items retrieved at least once.
            - Tuple of lists containing per-query values for Hit@K, Recall@K,
              Precision@K, and MRR, in that order.
    """
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
    """
    Evaluate answer quality using LLM-as-a-judge for both faithfulness and correctness.

    This function calls `judge_faithfulness` and `judge_correctness` for each query–answer 
    pair, aggregates their results into overall metrics, and also returns the raw 
    per-query judgments for deeper analysis.

    Args:
        llm: LLM interface used for evaluation.
        prompt_dir (str): Directory containing Jinja2 templates for prompts.
        system_template_name (str): Template file for the system prompt.
        faithfulness_user_template_name (str): Template file for the user prompt in faithfulness evaluation.
        correctness_user_template_name (str): Template file for the user prompt in correctness evaluation.
        all_questions (List[str]): List of user queries being evaluated.
        all_answers (List[str]): Model-generated answers for each query.
        all_retrieved_texts (List[List[str]]): Retrieved context texts per query (for faithfulness).
        all_golden_texts (List[List[str]]): Golden reference texts per query (for correctness).

    Returns:
        Tuple[Dict[str, float], Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]]:
            - Aggregated evaluation metrics including:
                - "faithfulness_score": Mean LLM-judged faithfulness.
                - "correctness_score": Mean correctness score.
                - "hallucination_score": Proportion of "contradicted" verdicts.
                - "insufficient_rate_faith": Proportion of "insufficient" verdicts in faithfulness.
                - "insufficient_rate_correct": Proportion of "insufficient" verdicts in correctness.
                - "abstain_accuracy_no_gold": Accuracy of abstains when no gold exists.
            - Tuple of lists containing raw faithfulness and correctness results per query.
    """
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
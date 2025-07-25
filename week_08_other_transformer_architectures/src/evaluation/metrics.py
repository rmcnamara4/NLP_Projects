from rouge_score import rouge_scorer
from collections import defaultdict
import numpy as np

def compute_rouge_scores(predictions_dict, references_dict):
    """
    Computes average ROUGE scores between predicted and reference summaries.

    Args:
        predictions_dict (dict): Dictionary mapping ID to predicted summary.
        references_dict (dict): Dictionary mapping ID to reference summary.

    Returns:
        dict: Average ROUGE-1, ROUGE-2, and ROUGE-L scores.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer = True)
    scores = defaultdict(list)

    for aid, pred in predictions_dict.items():
        ref = references_dict.get(aid)
        if ref is None:
            continue  # skip if reference is missing for that ID

        result = scorer.score(ref, pred)
        for key in result:
            scores[key].append(result[key].fmeasure)

    avg_scores = {k: np.mean(v) for k, v in scores.items()}
    return avg_scores
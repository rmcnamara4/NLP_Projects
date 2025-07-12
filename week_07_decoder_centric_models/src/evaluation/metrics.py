from rouge_score import rouge_scorer
from collections import defaultdict
import numpy as np

def compute_rouge_scores(predictions_dict, references_list):
    """
    Computes average ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L) between generated summaries and reference summaries.

    This function uses the `rouge_scorer` from the `rouge_score` package to calculate F1 scores for each ROUGE metric
    on a per-sample basis, and then averages them across all examples.

    Args:
        predictions_dict (Dict[int, str]): A dictionary mapping example IDs (or indices) to generated summaries.
        references_list (List[str]): A list of reference summaries in the same order as the example indices.

    Returns:
        Dict[str, float]: A dictionary containing the average F1 scores for 'rouge1', 'rouge2', and 'rougeL'.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer = True)

    scores = defaultdict(list)

    for idx, reference in enumerate(references_list):
        # Score
        combined_pred = predictions_dict[idx]
        result = scorer.score(reference, combined_pred)
        for key in result:
            scores[key].append(result[key].fmeasure)

    avg_scores = {k: np.mean(v) for k, v in scores.items()}
    return avg_scores
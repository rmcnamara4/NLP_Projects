from rouge_score import rouge_scorer
from collections import defaultdict
import numpy as np

def compute_rouge_scores(predictions_dict, references_list):
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
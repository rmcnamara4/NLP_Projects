from src.llm.prompting import render_template
from typing import List, Dict, Any
from collections import defaultdict
import numpy as np 

def run_judge(llm, system_prompt: str, user_prompt: str, **overrides) -> Dict[str, Any]: 
    messages = [
        {'role': 'system', 'content': system_prompt}, 
        {'role': 'user', 'content': user_prompt}
    ]

    resp = llm.generate(messages, **overrides) 
    raw = resp.get('text', '') or ''

    try:
        out = json.loads(raw) 
    except Exception: 
        out = {'verdict': 'insufficient', 'score': 0.0, 'rationale': 'Non-JSON output.'}

    out['verdict'] = str(out.get('verdict', 'insufficient')).lower()
    try: 
        out['score'] = float(out.get('score', 0.0))
    except Exception: 
        out['score'] = 0.0
    out['rationale'] = str(out.get('rationale', '')).strip()[:500]

    return out

def judge_faithfulness(
    llm, 
    prompt_dir: str, 
    system_template_name: str, 
    user_template_name: str, 
    question: str, 
    answer: str, 
    retrieved_context: List[str], 
    **overrides
) -> Dict[str, Any]: 
    system_prompt = render_template(prompt_dir, system_template_name) 
    user_prompt = render_template(
        prompt_dir, 
        user_template_name,
        question = question,
        answer = answer, 
        retrieved_context = retrieved_context or []
    )

    return run_judge(llm, system_prompt, user_prompt, **overrides) 

def judge_correctness(
    llm, 
    prompt_dir: str, 
    system_template_name: str, 
    user_template_name: str, 
    question: str, 
    answer: str, 
    golden_context: List[str], 
    **overrides
) -> Dict[str, Any]: 
    system_prompt = render_template(prompt_dir, system_template_name)
    user_prompt = render_template(
        prompt_dir, 
        user_template_name, 
        question = question, 
        answer = answer, 
        golden_context = golden_context or []
    )

    return run_judge(llm, system_prompt, user_prompt, **overrides)

def aggregate_judge_results(faithfulness_results: List[Dict[str, Any]], correctness_results: List[Dict[str, Any]], has_gold_flags: List[bool]) -> Dict[str, Any]: 
    metrics = defaultdict(float) 
    n = len(faithfulness_results)

    f_scores = [r.get('score', 0.0) for r in faithfulness_results]
    f_verdicts = [r.get('verdict', 'insufficient') for r in faithfulness_results]

    c_scores = [c.get('score', 0.0) for c in correctness_results]
    c_verdicts = [c.get('verdict', 'insufficient') for c in correctness_results]

    metrics['faithfulness_score'] = float(np.mean(f_scores)) if f_scores else 0.0 
    metrics['correctness_score'] = float(np.mean(c_scores)) if c_scores else 0.0 
    metrics['hallucination_score'] = float(np.mean([v == 'contradicted' for v in f_verdicts])) if n else 0.0 
    metrics['insufficient_rate_faith'] = float(np.mean([v == 'insufficient' for v in f_verdicts])) if n else 0.0 
    metrics['insufficient_rate_correct'] = float(np.mean([v == 'insufficient' for v in c_verdicts])) if n else 0.0 

    no_gold_idxs = [i for i, hg in enumerate(has_gold_flags) if not hg]
    if no_gold_idxs: 
        abstains = [c_verdicts[i] == 'abstain' for i in no_gold_idxs]
        metrics['abstain_accuracy_no_gold'] = float(np.mean(abstains))
    else: 
        metrics['abstain_accuracy_no_gold'] = None

    return dict(metrics)




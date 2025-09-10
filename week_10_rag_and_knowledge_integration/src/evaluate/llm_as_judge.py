from src.llm.prompting import render_template
from typing import List, Dict, Any
from collections import defaultdict
import numpy as np 
import json
import re

def _extract_json(text: str) -> str:
    """
    Extract the first JSON object enclosed in braces from a text string.

    The function checks for JSON wrapped in Markdown code fences (```json ... ```), 
    falling back to the first `{...}` block if found. If no braces are present, 
    returns the original text.

    Args:
        text (str): Input string that may contain a JSON object or code block.

    Returns:
        str: Extracted JSON substring if found, otherwise the original text.
    """
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags = re.S)
    if m:
        return m.group(1)

    m = re.search(r"\{.*?\}", text, flags = re.S)
    return m.group(0) if m else text

def run_judge(llm, system_prompt: str, user_prompt: str, **overrides) -> Dict[str, Any]: 
    """
    Run an LLM-based judging step with system and user prompts, returning a
    normalized JSON verdict.

    The function calls the provided LLM with the given prompts, extracts a JSON
    object from its response, and ensures standard fields (`verdict`, `score`,
    `rationale`) are present and formatted consistently.

    Args:
        llm: An LLM interface with a `.generate(messages, **overrides)` method.
        system_prompt (str): Instructional text for the system role.
        user_prompt (str): Task or evaluation prompt for the user role.
        **overrides: Optional generation parameters (e.g., temperature, max_tokens).

    Returns:
        Dict[str, Any]: A JSON-like dictionary with keys:
            - "verdict" (str): One of {"supported", "contradicted", "insufficient", "abstain"}.
            - "score" (float): Normalized score (default 0.0 if parsing fails).
            - "rationale" (str): Short explanation (max 500 chars).
    """
    messages = [
        {'role': 'system', 'content': system_prompt}, 
        {'role': 'user', 'content': user_prompt}
    ]

    resp = llm.generate(messages, **overrides) 
    raw = _extract_json(resp.get('text', '') or '')

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
    """
    Evaluate whether an answer is faithful to the retrieved context using an LLM-as-judge.

    This function renders system and user prompts from templates, incorporating the
    question, answer, and retrieved evidence, then passes them to the LLM judge.
    The LLM is expected to return a JSON verdict with fields like "verdict", "score",
    and "rationale".

    Args:
        llm: LLM interface with a `.generate(messages, **overrides)` method.
        prompt_dir (str): Directory path containing Jinja2 templates.
        system_template_name (str): Filename of the system prompt template.
        user_template_name (str): Filename of the user prompt template.
        question (str): Original user question being answered.
        answer (str): Candidate answer to evaluate for faithfulness.
        retrieved_context (List[str]): Retrieved evidence passages to ground the answer.
        **overrides: Optional LLM generation parameters (e.g., temperature, max_tokens).

    Returns:
        Dict[str, Any]: LLM judgment dictionary with standardized keys:
            - "verdict" (str): Supported, contradicted, insufficient, or abstain.
            - "score" (float): Faithfulness score in [0,1].
            - "rationale" (str): Short justification for the verdict.
    """
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
    """
    Evaluate whether an answer is correct with respect to the golden context using an LLM-as-judge.

    This function renders system and user prompts from templates, incorporating the
    question, answer, and ground-truth (golden) evidence, then passes them to the LLM judge.
    The LLM is expected to return a JSON verdict with fields like "verdict", "score",
    and "rationale".

    Args:
        llm: LLM interface with a `.generate(messages, **overrides)` method.
        prompt_dir (str): Directory path containing Jinja2 templates.
        system_template_name (str): Filename of the system prompt template.
        user_template_name (str): Filename of the user prompt template.
        question (str): Original user question being answered.
        answer (str): Candidate answer to evaluate for correctness.
        golden_context (List[str]): Ground-truth evidence passages to verify correctness.
        **overrides: Optional LLM generation parameters (e.g., temperature, max_tokens).

    Returns:
        Dict[str, Any]: LLM judgment dictionary with standardized keys:
            - "verdict" (str): Supported, contradicted, insufficient, or abstain.
            - "score" (float): Correctness score in [0,1].
            - "rationale" (str): Short justification for the verdict.
    """
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
    """
    Aggregate evaluation results from LLM-as-judge outputs into summary metrics.

    This function computes average scores and rates across multiple queries for both
    faithfulness (consistency with retrieved context) and correctness (alignment with
    golden evidence). It also tracks hallucination rate, insufficient evidence rates,
    and abstain accuracy for cases without gold context.

    Args:
        faithfulness_results (List[Dict[str, Any]]): List of verdict dictionaries from
            `judge_faithfulness`, each containing "score" and "verdict".
        correctness_results (List[Dict[str, Any]]): List of verdict dictionaries from
            `judge_correctness`, each containing "score" and "verdict".
        has_gold_flags (List[bool]): Flags indicating which queries had gold evidence
            available (True if gold evidence exists, False otherwise).

    Returns:
        Dict[str, Any]: Dictionary of aggregated metrics:
            - "faithfulness_score" (float): Mean faithfulness score.
            - "correctness_score" (float): Mean correctness score.
            - "hallucination_score" (float): Proportion of faithfulness verdicts labeled "contradicted".
            - "insufficient_rate_faith" (float): Proportion of faithfulness verdicts labeled "insufficient".
            - "insufficient_rate_correct" (float): Proportion of correctness verdicts labeled "insufficient".
            - "abstain_accuracy_no_gold" (float | None): Accuracy of abstain verdicts on queries
              without gold evidence; None if all queries had gold.
    """
    metrics = defaultdict(float) 
    n = len(faithfulness_results)

    f_scores = [r.get('score', 0.0) for r in faithfulness_results]
    f_verdicts = [r.get('verdict', 'insufficient') for r in faithfulness_results]

    c_scores = [c.get('score', 0.0) for c in correctness_results]
    c_verdicts = [c.get('verdict', 'insufficient') for c in correctness_results]

    metrics['faithfulness_score'] = float(np.nanmean(f_scores)) if f_scores else 0.0 
    metrics['correctness_score'] = float(np.nanmean(c_scores)) if c_scores else 0.0 
    metrics['hallucination_score'] = float(np.nanmean([v == 'contradicted' for v in f_verdicts])) if n else 0.0 
    metrics['insufficient_rate_faith'] = float(np.nanmean([v == 'insufficient' for v in f_verdicts])) if n else 0.0 
    metrics['insufficient_rate_correct'] = float(np.nanmean([v == 'insufficient' for v in c_verdicts])) if n else 0.0 

    no_gold_idxs = [i for i, hg in enumerate(has_gold_flags) if not hg]
    if no_gold_idxs: 
        abstains = [c_verdicts[i] == 'abstain' for i in no_gold_idxs]
        metrics['abstain_accuracy_no_gold'] = float(np.nanmean(abstains))
    else: 
        metrics['abstain_accuracy_no_gold'] = None

    return dict(metrics)




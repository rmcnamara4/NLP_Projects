import json 
import argparse

from src.evaluate.pipeline import * 
from src.retrieval.pipeline import load_index
from src.llm.bedrock_llm import BedrockLLM
from src.utils.runlog import save_runlog

import pandas as pd 
import os

def main(): 
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '--queries_path', 
        default = 'data/queries.jsonl'
    )
    ap.add_argument(
        '--retrieved_path', 
        default = 'data/retrieved_chunks.jsonl'
    )
    ap.add_argument(
        '--answer_path', 
        default = 'results/rag_answers.jsonl'
    )
    ap.add_argument(
        '--model_id', 
        default = 'us.anthropic.claude-3-5-haiku-20241022-v1:0'
    )
    ap.add_argument(
        '--temperature',
        type = float, 
        default = 0.0
    )
    ap.add_argument(
        '--top_p', 
        type = float, 
        default = 1.0 
    )
    ap.add_argument(
        '--max_tokens', 
        type = int, 
        default = 600
    )
    ap.add_argument(
        '--prompt_dir', 
        default = 'prompts/version_1/eval'
    )
    ap.add_argument(
        '--system_template', 
        default = 'system.j2'
    )
    ap.add_argument(
        '--faithfulness_user_template', 
        default = 'faithfulness_user.j2'
    )
    ap.add_argument(
        '--correctness_user_template', 
        default = 'correctness_user.j2'
    )
    ap.add_argument(
        '--processed_chunks_path', 
        default = 'data/processed/processed_chunks.jsonl'
    )
    ap.add_argument(
        '--metadata_path', 
        default = 'index/faiss_store_v1/metadatas.jsonl'
    )
    ap.add_argument(
        '--index_path', 
        default = 'index/faiss_store_v1/index.faiss'
    )
    ap.add_argument(
        '--k', 
        type = int,
        default = 5
    )
    ap.add_argument(
        '--match_type', 
        choices = ['binary', 'semantic'],
        default = 'binary'
    )
    ap.add_argument(
        '--sim_threshold', 
        type = float, 
        default = None
    )
    ap.add_argument(
        '--out', 
        default = 'results'
    )
    args = ap.parse_args()

    save_runlog(args, 'evaluate_rag')

    #########################################################################################
    # Load Data 
    #########################################################################################
    retrieved_ids = get_retrieved_ids(
        args.retrieved_path, 
        use_s3 = False
    )
    golden_ids = get_golden_ids(
        args.queries_path, 
        use_s3 = False
    )
    questions, answers = get_questions_and_answers(
        args.answer_path, 
        use_s3 = False
    )

    retrieved_texts = get_text(retrieved_ids, args.processed_chunks_path, use_s3 = True)
    golden_texts = get_text(golden_ids, args.processed_chunks_path, use_s3 = True) 

    #########################################################################################
    # Evaluate retrieval mechanism 
    #########################################################################################
    print('Performing retrieval evaluation!') 

    if args.match_type == 'semantic': 
        index = load_index(args.index_path) 
        get_meta_to_int = get_meta_to_int(args.metadata_path, use_s3 = False)

        retrieved_vecs = get_vecs_from_ids(retrieved_ids, index.index, meta_to_int)
        golden_vecs = get_vecs_from_ids(golden_ids, index.index, meta_to_int)

        retrieval_results, (hit_at_k, recall_at_k, precision_at_k, mrr) = eval_retrieval(retrieved_vecs, golden_vecs, k = args.k, match_type = args.match_type, sim_threshold = args.sim_threshold) 

    else: 
        retrieval_results, (hit_at_k, recall_at_k, precision_at_k, mrr) = eval_retrieval(retrieved_ids, golden_ids, k = args.k, match_type = args.match_type, sim_threshold = args.sim_threshold) 

    print('Finished retrieval evaluation!')

    #########################################################################################
    # Evaluate correctness / faithfulness with LLM as a judge 
    #########################################################################################
    print('Starting LLM as a judge evaluation!')

    llm = BedrockLLM(
        model_id = args.model_id, 
        default = {
            'temperature': args.temperature, 
            'top_p': args.top_p, 
            'max_tokens': args.max_tokens
        }
    )

    judge_metrics, (faithfulness_results, correctness_results) = eval_correctness_and_faithfulness(
        llm, 
        args.prompt_dir, 
        args.system_template, 
        args.faithfulness_user_template, 
        args.correctness_user_template, 
        questions, 
        answers, 
        retrieved_texts, 
        golden_texts
    )

    print('Finished LLM as a judge evaluation!')

    #########################################################################################
    # Save
    #########################################################################################
    os.makedirs(args.out, exist_ok = True)

    correctness_df = pd.DataFrame(correctness_results) 
    correctness_df.columns = ['correctness_verdict', 'correctness_score', 'correctness_rationale']

    faithfulness_df = pd.DataFrame(faithfulness_results) 
    faithfulness_df.columns = ['faithfulness_verdict', 'faithfulness_score', 'faithfulness_rationale']

    retrieval_df = pd.DataFrame({
        'id': ['q' + str(i) for i in range(1, len(questions) + 1)],
        'question': questions, 
        'answer': answers,
        f'hit_at_{args.k}': hit_at_k, 
        f'recall_at_{args.k}': recall_at_k, 
        f'precision_at_{args.k}': precision_at_k, 
        'mrr': mrr
    })

    full_results = pd.concat((retrieval_df, correctness_df, faithfulness_df), axis = 1)
    full_results.to_csv(os.path.join(args.out, f'results_by_query_{args.match_type}.csv'), index = False, header = True)

    with open(os.path.join(args.out, f'retrieval_results_{args.match_type}.json'), 'w') as f: 
        json.dump(retrieval_results, f, indent = 2)
    
    with open(os.path.join(args.out, 'llm_judge_results.json'), 'w') as f: 
        json.dump(judge_metrics, f, indent = 2)

    print('Saved results!')

if __name__ == '__main__': 
    main()



    



    



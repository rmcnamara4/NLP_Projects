import json 
import argparse 
from pathlib import Path 

from src.llm.bedrock_llm import BedrockLLM
from src.llm.pipeline import answer_with_rag
from src.utils.io import load_jsonl, save_jsonl
from src.utils.runlog import save_runlog

def main(): 
    ap = argparse.ArgumentParser()
    ap.add_argument('--chunks_json', required = True)
    ap.add_argument('--retrieval_json', required = True) 
    ap.add_argument('--prompt_dir', default = 'prompts/version_1')
    ap.add_argument('--use_s3', action = 'store_true') 
    ap.add_argument('--model_id', default = 'us.anthropic.claude-3-5-haiku-20241022-v1:0')
    ap.add_argument('--max_context_char', type = int, default = 6000)
    ap.add_argument('--max_out_tokens', type = int, default = 700) 
    ap.add_argument('--temperature', type = float, default = 0.2) 
    ap.add_argument('--top_p', type = float, default = 0.95)
    ap.add_argument('--out', default = './results/rag_answers.jsonl')
    args = ap.parse_args()

    save_runlog(args, sub_dir = 'run_rag')

    queries = load_jsonl(args.retrieval_json, use_s3 = False) 
    
    llm = BedrockLLM(
        model_id = args.model_id, 
        default = {
            'max_tokens': args.max_out_tokens, 
            'temperature': args.temperature, 
            'top_p': args.top_p
        }
    )
    
    outputs = []
    for q in queries: 
        out = answer_with_rag(
            question = q['query'], 
            retrieved_results = q['results'], 
            chunks_json_path = args.chunks_json, 
            use_s3 = args.use_s3, 
            max_chars = args.max_context_char, 
            prompt_dir = args.prompt_dir,
            llm = llm
        )
        out['id'] = q.get('id') 
        outputs.append(out) 

    save_jsonl(outputs, args.out, use_s3 = False) 
    print(f'Wrote {len(outputs)} answers -> {args.out}')

if __name__ == '__main__': 
    main()
    
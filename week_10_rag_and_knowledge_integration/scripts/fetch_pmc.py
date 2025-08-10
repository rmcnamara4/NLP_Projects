import argparse
from src.data.pipeline import run_pipeline

if __name__ == '__main__': 
    ap = argparse.ArgumentParser()
    ap.add_argument('--query', required = True)
    ap.add_argument('--max_results', type = int, default = 50)
    ap.add_argument('--date_from', default = None)
    ap.add_argument('--date_to', default = None)
    ap.add_argument('--use_s3', action = 'store_true')
    args = ap.parse_args()

    run_pipeline(
        query = args.query, 
        max_results = args.max_results, 
        date_from = args.date_from, 
        date_to = args.date_to, 
        use_s3 = args.use_s3, 
        raw_key = f'data/raw/raw_xml.jsonl', 
        interim_key = f'data/interim/parsed_xml.jsonl', 
        processed_key = f'data/processed/article_chunks.jsonl'
    )



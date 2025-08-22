import argparse
from src.data.pipeline import collect_data
from src.utils.runlog import save_runlog
import os
from datetime import datetime
import json

if __name__ == '__main__': 
    ap = argparse.ArgumentParser()
    ap.add_argument('--query', required = True)
    ap.add_argument('--max_results', type = int, default = 50)
    ap.add_argument('--date_from', default = None)
    ap.add_argument('--date_to', default = None)
    ap.add_argument('--use_s3', action = 'store_true')
    ap.add_argument(
        '--raw_key', 
        default = 'data/raw/raw_xml.jsonl'
    )
    ap.add_argument(
        '--interim_key',
        default = 'data/interim/parsed_xml.jsonl'
    )
    args = ap.parse_args()

    save_runlog(args, sub_dir = 'fetch_pmc')

    collect_data(
        query = args.query, 
        max_results = args.max_results, 
        date_from = args.date_from, 
        date_to = args.date_to, 
        use_s3 = args.use_s3, 
        raw_key = args.raw_key, 
        interim_key = args.interim_key
    )



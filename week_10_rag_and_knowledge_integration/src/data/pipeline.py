from typing import List, Optional
from src.data.pmc_client import * 
from src.data.pmc_parser import * 
from src.data.io import * 

def run_pipeline(
    query: str, 
    max_results: int, 
    date_from: Optional[str], 
    date_to: Optional[str], 
    save_raw_xml: bool = True, 
    save_interim: bool = True, 
    save_processed: bool = True, 
    use_s3: bool = True, 
    raw_key: str = 'data/raw/pmc_raw.jsonl', 
    interim_key: str = 'data/interim/pmc_parsed.jsonl', 
    processed_key: str = 'data/processed/pmc_chunks.jsonl'
): 
    pmc_ids = search_pmc_ids(query, max_results, date_from, date_to)
    if not pmc_ids: 
        print('No results.') 
        return 
    
    xml_articles = fetch_pmc_articles(pmc_ids, retmode = 'xml') 

    if save_raw_xml: 
        save_jsonl([{'pmcid': pmcid, 'xml': xml} for pmcid, xml in zip(pmc_ids, xml_articles)], raw_key, use_s3)
        
    parsed = parse_many(xml_articles) 

    if save_interim: 
        save_jsonl(parsed, interim_key, use_s3)

    print('Done saving articles!') 

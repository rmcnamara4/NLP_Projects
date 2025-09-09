from typing import List, Optional
from src.data.pmc_client import * 
from src.data.pmc_parser import * 
from src.utils.io import * 

def collect_data(
    query: str, 
    max_results: int, 
    date_from: Optional[str], 
    date_to: Optional[str], 
    save_raw_xml: bool = True, 
    save_interim: bool = True, 
    use_s3: bool = True, 
    raw_key: str = 'data/raw/pmc_raw.jsonl', 
    interim_key: str = 'data/interim/pmc_parsed.jsonl', 
): 
    """
    Collects articles from PubMed Central (PMC) based on a search query and saves them.

    This function searches for PMC IDs using the given query and date range,
    fetches corresponding articles, optionally saves the raw XML and parsed JSONL
    data to local storage or S3, and prints a completion message.

    Workflow:
        1. Search PMC for matching IDs using `search_pmc_ids`.
        2. Fetch full-text XML articles using `fetch_pmc_articles`.
        3. Optionally save raw XML content as JSONL (one record per article).
        4. Parse XML articles with `parse_many` to extract metadata/content.
        5. Optionally save parsed JSONL data for downstream use.

    Args:
        query (str):
            Search query string for PMC (e.g., "Alzheimer's disease").
        max_results (int):
            Maximum number of articles to fetch.
        date_from (Optional[str]):
            Start date for filtering articles (YYYY or YYYY/MM/DD). If None, no lower bound.
        date_to (Optional[str]):
            End date for filtering articles (YYYY or YYYY/MM/DD). If None, no upper bound.
        save_raw_xml (bool, optional):
            Whether to save raw XML articles as JSONL. Defaults to True.
        save_interim (bool, optional):
            Whether to save parsed metadata as JSONL. Defaults to True.
        use_s3 (bool, optional):
            If True, saves to S3 using the provided keys. If False, saves locally. Defaults to True.
        raw_key (str, optional):
            Path or S3 key for saving raw XML JSONL. Defaults to "data/raw/pmc_raw.jsonl".
        interim_key (str, optional):
            Path or S3 key for saving parsed JSONL. Defaults to "data/interim/pmc_parsed.jsonl".

    Returns:
        None

    Notes:
        - If no results are found, the function prints "No results." and exits.
        - Relies on helper functions: `search_pmc_ids`, `fetch_pmc_articles`,
          `parse_many`, and `save_jsonl`.
    """
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

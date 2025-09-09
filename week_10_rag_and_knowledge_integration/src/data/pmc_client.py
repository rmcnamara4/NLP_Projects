import time, os 
from typing import List, Optional, Dict
from Bio import Entrez 
from urllib.error import HTTPError

import xml.etree.ElementTree as ET

import os
from dotenv import load_dotenv
load_dotenv()

EMAIL = os.getenv('PUBMED_EMAIL')
Entrez.email = EMAIL

def search_pmc_ids(
    query: str,
    max_results: int = 50,
    date_from: Optional[str] = None,   
    date_to: Optional[str] = None      
) -> List[str]:
    """
    Searches PubMed Central (PMC) for article IDs using the NCBI Entrez API.

    This function issues an ESearch query against the PMC database and returns
    a list of matching article identifiers in the format "PMC####...".

    Args:
        query (str):
            Search term or query string to run against PMC.
        max_results (int, optional):
            Maximum number of results to retrieve. Defaults to 50.
        date_from (Optional[str], optional):
            Lower bound for publication date filter (YYYY or YYYY/MM/DD). 
            If None, no lower bound is applied.
        date_to (Optional[str], optional):
            Upper bound for publication date filter (YYYY or YYYY/MM/DD).
            If None, no upper bound is applied.

    Returns:
        List[str]: A list of PMC article IDs (e.g., ["PMC1234567", "PMC7654321"]).
    """
    esearch_kwargs = {
        'db': 'pmc',
        'term': query,          
        'retmax': max_results,
    }
    if date_from or date_to:
        esearch_kwargs.update({
            'datetype': 'pdat',
            'mindate': date_from or '1000',
            'maxdate': date_to or '3000',
        })

    with Entrez.esearch(**esearch_kwargs) as h:
        rec = Entrez.read(h)

    ids = rec.get('IdList', [])

    return [f'PMC{id_}' for id_ in ids]

def fetch_pmc_articles(
    pmc_ids: List[str], 
    retmode: str = 'xml'
) -> List[str]: 
    """
    Fetches full-text articles from PubMed Central (PMC) using the NCBI Entrez API.

    Given a list of PMC identifiers, this function retrieves the corresponding
    articles in the requested return mode.

    Args:
        pmc_ids (List[str]):
            A list of PMC IDs (e.g., ["PMC1234567", "PMC7654321"]) to fetch.
        retmode (str, optional):
            The return mode. Defaults to "xml".
            - "xml": Returns a list of XML strings, one per <article>.
            - Other values: Returns the raw data as a single string.

    Returns:
        List[str]:
            - If `retmode="xml"`: A list of XML article strings.
            - If `retmode` is another format: A single-element list containing the raw data.
              Returns an empty list if no IDs are provided or if parsing fails.

    Notes:
        - Uses `Entrez.efetch` under the hood.
        - XML parsing errors will result in an empty list.
    """
    if not pmc_ids: 
        return []
    
    id_str = ' '.join(pmc_ids) 
    handle = Entrez.efetch(db = 'pmc', id = id_str, retmode = retmode) 
    data = handle.read()
    handle.close()

    if retmode == 'xml': 
        try: 
            root = ET.fromstring(data) 
            return [ET.tostring(article, encoding = 'unicode') for article in root.findall('.//article')]
        except ET.ParseError: 
            return []
    else: 
        return data
    

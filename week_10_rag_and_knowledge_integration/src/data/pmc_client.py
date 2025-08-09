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
    

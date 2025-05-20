from Bio import Entrez, Medline 
import pandas as pd 
import numpy as np 
import time 

def search_abstracts(query, max_results = 100_000):
  """
  Search PubMed for article IDs using a specified query.

  Parameters:
      query (str): The search term or MeSH query to run against PubMed.
      max_results (int, optional): Maximum number of PubMed IDs to retrieve. Default is 100,000.

  Returns:
      list: A list of PubMed IDs (PMIDs) matching the query.
  """
  handle = Entrez.esearch(db = 'pubmed', term = query, retmax = max_results, retmode = 'xml')
  record = Entrez.read(handle)
  return record['IdList']

def fetch_abstracts(ids, batch_size = 500):
  """
  Fetch PubMed article metadata (title, abstract, PMID) for a list of IDs.

  Parameters:
      ids (list): A list of PubMed IDs (PMIDs) to retrieve data for.
      batch_size (int, optional): Number of articles to fetch per API call. Default is 500.

  Returns:
      list: A list of dictionaries, each containing 'PMID', 'Title', and 'Abstract' for one article.
  """
  abstracts = []
  for i in range(0, len(ids), batch_size):
    batch_ids = ids[i:i + batch_size]
    fetch_handle = Entrez.efetch(db = 'pubmed', id = ','.join(batch_ids), rettype = 'medline', retmode = 'text')
    records = Medline.parse(fetch_handle)
    for rec in records:
      title = rec.get('TI', 'No Title')
      abstract = rec.get('AB', 'No Abstract')
      id = rec.get('PMID', 'Unknown')
      abstracts.append({'PMID': id, 'Title': 'title', 'Abstract': abstract})

    print(f'Fetched {i + len(batch_ids)} abstracts so far...')
    time.sleep(1)

  return abstracts
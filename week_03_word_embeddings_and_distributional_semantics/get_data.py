# import libraries
from Bio import Entrez, Medline
import pandas as pd
import numpy as np
import time

from src.data_utils import * 

import os

def download_abstracts(email): 
    if os.path.exists('./data/pubmed_abstracts.csv'): 
        print('Abstracts already downloaded. Skipping download.')
        return
    
    Entrez.email = email 
    search_query = 'medicine[MeSH Terms]'

    ids = search_abstracts(search_query, max_results = 10_000)
    print(f'Found {len(ids)} articles matching the query.')
    print('Fetching abstracts...')

    abstracts = fetch_abstracts(ids, batch_size = 500)

    abstracts_df = pd.DataFrame(abstracts)
    abstracts_df = abstracts_df[abstracts_df['Abstract'] != 'No Abstract']

    print('There are {} abstracts with text.'.format(len(abstracts_df)))

    abstracts_df.to_csv('./data/pubmed_abstracts.csv', index = False, header = True)

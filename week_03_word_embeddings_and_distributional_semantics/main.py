# import libraries
from Bio import Entrez, Medline
import pandas as pd
import numpy as np
import time

from get_data import download_abstracts

import os

email = os.getenv('PUBMED_EMAIL')
if not email: 
    raise ValueError('Please set the PUBMED_EMAIL environment variable to your email address.') 

if __name__ == '__main__':
    download_abstracts(email)


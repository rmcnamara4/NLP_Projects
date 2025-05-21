# import libraries
from Bio import Entrez, Medline
import pandas as pd
import numpy as np
import time

from get_data import download_abstracts
from src.preprocessing_utils import * 
from src.plot_utils import * 
from src.evaluation_utils import * 
from src.model_utils import *

import swifter 
from umap import UMAP

from tokenizers import Tokenizer
from tokenizers.models import BPE   
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

from datasets import load_dataset

from sklearn.metrics.pairwise import cosine_similarity

import os

email = os.getenv('PUBMED_EMAIL')
if not email: 
    raise ValueError('Please set the PUBMED_EMAIL environment variable to your email address.') 

ANALOGY_QUERIES = [
    (['cardiovascular', 'lung'], ['heart']),
    (['antibiotic', 'virus'], ['bacteria']),
    (['mental', 'cardiology'], ['psychiatry']),
    (['oncology', 'pneumonia'], ['cancer']),
]

SIMILARITY_WORDS = ['diabetes', 'cancer', 'disease', 'analysis', 'cardiovascular']

if __name__ == '__main__':
    # download_abstracts(email)

    data = pd.read_csv('./data/pubmed_abstracts.csv') 
    abstracts = data['Abstract'].dropna() 

    ###############################################################
    # Full Word Skip-Gram 
    ###############################################################
    # Preprocess the abstracts
    tokenized_abstracts = abstracts.swifter.apply(clean_text, bpe = False)

    # Train and save model 
    print('Training Skip-Gram model...')
    skipgram_model = train_model(
        vector_size = 200,
        window = 5, 
        sg = 1, 
        min_count = 20, 
        workers = 4, 
        tokenized_abstracts = tokenized_abstracts.tolist(), 
        epochs = 10
    )
    skipgram_model.save('./results/skipgram/skipgram_model.model')
                        
    embeddings = skipgram_model.wv 
    embeddings.save('./results/skipgram/skipgram.embeddings') 

    # visualize the embeddings 
    # all words in the vocab 
    words = list(embeddings.index_to_key) 
    vectors = embeddings[words]

    reducer = UMAP(n_neighbors = 15, min_dist = 0.1, metric = 'cosine', random_state = 42)
    embeddings_2d = reducer.fit_transform(vectors) 

    plot_umap(embeddings_2d, words, alpha = 0.3, s = 10, save_path = './results/skipgram/umap_full_vocab.png') 

    # most common words 
    common_words = words[:3000]
    common_vectors = embeddings[common_words]

    reducer = UMAP(n_neighbors = 15, min_dist = 0.1, metric = 'cosine', random_state = 42)
    common_embeddings_2d = reducer.fit_transform(common_vectors)

    plot_umap(common_embeddings_2d, common_words, alpha = 0.3, s = 10, save_path = './results/skipgram/umap_common_words.png')

    # evaluate similarity 
    get_n_similar_words(
        SIMILARITY_WORDS, 
        embeddings, 
        n = 5, 
        save_path = './results/skipgram/similar_words_output.csv'
    )

    # plot cosine similarity histogram 
    similarities = cosine_similarity(vectors, vectors)
    upper_i, upper_j = np.triu_indices(similarities.shape[0], k = 1)
    similarities = similarities[upper_i, upper_j]

    plot_similarity_hist(similarities, bins = 30, kde = False, save_path = './results/skipgram/cosine_similarity_histogram.png')

    # evaluate analogy 
    get_analogies(
        ANALOGY_QUERIES, 
        embeddings, 
        n = 5, 
        save_path = './results/skipgram/analogy_results.csv'
    )

    ###############################################################
    # BPE Skip-Gram
    ###############################################################
    # clean and tokenize the abstracts 
    cleaned_text = abstracts.swifter.apply(lambda x: ' '.join(clean_text(x, bpe = True)))
    cleaned_text.to_csv('./data/bpe_skipgram/cleaned_abstracts.csv', index = False, header = True)

    print('Training BPE tokenizer...')
    tokenizer = Tokenizer(BPE(unk_token = '<UNK>'))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size = 10_000, 
        special_tokens = ['<UNK>', '<NUM>', '<PAD>']
    )
    tokenizer.train(['./data/bpe_skipgram/cleaned_abstracts.csv'], trainer = trainer) 
    tokenizer.save('./data/bpe_skipgram/bpe_tokenizer.json') 

    bpe_tokenized_text = cleaned_text.squeeze().swifter.apply(lambda x: tokenizer.encode(x).tokens).tolist()

    # train model 
    print('Training BPE Skip-Gram model...')
    bpe_skipgram_model = train_model(
        vector_size = 200,
        window = 5, 
        sg = 1, 
        min_count = 20, 
        workers = 4, 
        tokenized_abstracts = bpe_tokenized_text,
        epochs = 10
    )
    bpe_skipgram_model.save('./results/bpe_skipgram/bpe_model.model')

    bpe_embeddings = bpe_skipgram_model.wv
    bpe_embeddings.save('./results/bpe_skipgram/bpe_skipgram.embeddings')

    # visulalize the embeddings
    # all words in the vocab
    bpe_tokens = list(bpe_embeddings.index_to_key)
    bpe_vectors = bpe_embeddings[bpe_tokens]

    reducer = UMAP(n_neighbors = 15, min_dist = 0.1, metric = 'cosine', random_state = 42)
    bpe_embeddings_2d = reducer.fit_transform(bpe_vectors)

    plot_umap(bpe_embeddings_2d, bpe_tokens, alpha = 0.3, s = 10, save_path = './results/bpe_skipgram/umap_full_vocab.png')

    # most common words 
    bpe_common_tokens = bpe_tokens[:3000]
    bpe_common_vectors = bpe_embeddings[bpe_common_tokens]

    reducer = UMAP(n_neighbors = 15, min_dist = 0.1, metric = 'cosine', random_state = 42)
    bpe_common_embeddings_2d = reducer.fit_transform(bpe_common_vectors)

    plot_umap(bpe_common_embeddings_2d, bpe_common_tokens, alpha = 0.3, s = 10, save_path = './results/bpe_skipgram/umap_common_words.png')

    # evaluate similarity
    get_n_similar_words(
        SIMILARITY_WORDS, 
        bpe_embeddings, 
        n = 5, 
        save_path = './results/bpe_skipgram/similar_words_output.csv'
    )

    # plot cosine similarity histogram
    bpe_similarities = cosine_similarity(bpe_vectors, bpe_vectors)
    upper_i, upper_j = np.triu_indices(bpe_similarities.shape[0], k = 1)
    bpe_similarities = bpe_similarities[upper_i, upper_j]

    plot_similarity_hist(
        bpe_similarities, bins = 30, kde = False, save_path = './results/bpe_skipgram/cosine_similarity_histogram.png'
    )

    # evaluate analogy
    get_analogies(
        ANALOGY_QUERIES, 
        bpe_embeddings, 
        n = 5, 
        save_path = './results/bpe_skipgram/analogy_results.csv'
    )

    ###############################################################
    # FINAL EVALUATION 
    ###############################################################
    print('Evaluating models on UMNSRS dataset...')
    evaluation_data = load_dataset('bigbio/umnsrs', trust_remote_code = True)
    evaluation_data = evaluation_data['train'].to_pandas()
    evaluation_data = evaluation_data[['text_1', 'text_2', 'mean_score']]

    # evaluate on all terms 
    skip_gram_inds, skip_gram_results = evaluate(evaluation_data, embeddings, bpe_tokenizer = None)

    bpe_inds, bpe_results = evaluate(evaluation_data, bpe_embeddings, bpe_tokenizer = tokenizer) 

    # evaluate only on subset of terms 
    filtered_evaluation_data = evaluation_data.iloc[skip_gram_inds, :]

    _, skip_gram_results_filtered = evaluate(filtered_evaluation_data, embeddings, bpe_tokenizer = None) 

    _, bpe_results_filtered = evaluate(filtered_evaluation_data, bpe_embeddings, bpe_tokenizer = tokenizer)

    # save results 
    results_df = pd.DataFrame([
        ['Skip-Gram (Full Eval)', skip_gram_results[0], skip_gram_results[1], skip_gram_results[2]],
        ['BPE (Full Eval)', bpe_results[0], bpe_results[1], bpe_results[2]],
        ['Skip-Gram (Filtered)', skip_gram_results_filtered[0], skip_gram_results_filtered[1], skip_gram_results_filtered[2]],
        ['BPE (Filtered)', bpe_results_filtered[0], bpe_results_filtered[1], bpe_results_filtered[2]],
    ], columns=['Model', 'Spearman', 'Pearson', 'Num Evaluated'])

    results_df.to_csv('./results/evaluation_summary.csv', index = False, header = True)










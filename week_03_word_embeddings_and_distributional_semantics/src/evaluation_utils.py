import csv 
import numpy as np 
import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity 
from scipy.stats import spearmanr, pearsonr 

def get_n_similar_words(words, embeddings, n, save_path):
  """
  Prints the top-N most similar words to a given word based on cosine similarity.

  Args:
      word (str): The target word to find similar words for.
      embeddings (gensim KeyedVectors): Trained word embedding model (e.g., Word2Vec.wv).
      n (int): Number of most similar words to retrieve.
      save_path (str): if provided, the plot will be saved to this path.

  Returns:
      None: Saves similar words and their similarity scores to a csv file.
  """
  with open(save_path, 'w', newline = '') as f: 
    writer = csv.writer(f) 
    writer.writerow(['Target Word', 'Similar Word', 'Similarity'])
    for word in words: 
      similar_words = embeddings.most_similar(word, topn = n)
      for w, s in similar_words: 
        writer.writerow([word, w, round(s, 4)])

def get_analogies(analogy_queries, embeddings, n, save_path): 
    """
    Finds and prints the top-N most similar words based on analogy relationships.
    
    Args:
        analogy_queries (List[Tuple(List)]): List of tuples containing analogy queries in the form ([A, B], [C]) where 'C is to A as B is to ...'
        embeddings (gensim KeyedVectors): Trained word embedding model (e.g., Word2Vec.wv).
        n (int): Number of most similar words to retrieve.
        save_path (str): if provided, the plot will be saved to this path.
    
    Returns:
        None: Saves analogy results and top-n most similar words and similarities to a csv file.
    """
    with open(save_path, 'w', newline = '') as f: 
       writer = csv.writer(f)
       writer.writerow(['Analogy', 'Predicted Word', 'Similarity'])
       for positives, negatives in analogy_queries: 
          analogy_str = f"{negatives[0]} is to {positives[0]} as {positives[1]} is to ..."
          results = embeddings.most_similar(positives = positives, negatives = negatives, topn = n)
          for w, s in results: 
             writer.writerow([analogy_str, w, round(s, 4)])

def get_bpe_embedding(word, tokenizer, embeddings):
  """
  Retrieves the embedding for a given word using BPE subword embeddings.

  If the word exists in the embedding vocabulary, its embedding is returned directly.
  Otherwise, the word is tokenized into subwords using the BPE tokenizer, and the
  average of the available subword embeddings is returned.

  Args:
      word (str): The word to retrieve an embedding for.
      tokenizer (tokenizers.Tokenizer): A pretrained Hugging Face BPE tokenizer.
      embeddings (gensim KeyedVectors): Trained BPE word vector model.

  Returns:
      np.ndarray or None: The embedding vector for the word, or None if no tokens are found in the vocabulary.
  """
  if word in embeddings:
    return embeddings[word]
  else:
    tokens = tokenizer.encode(word).tokens
    vectors = [embeddings[token] for token in tokens if token in embeddings]
    if vectors:
      return np.mean(vectors, axis = 0)
    else:
      return None
    
def get_similarity(w1, w2, embeddings, bpe_tokenizer = None):
  """
  Computes the cosine similarity between two biomedical terms using either word-level
  or subword-level (BPE) embeddings.

  - If `bpe=True`, each word is converted to its BPE-based embedding using the average
    of its subword vectors (via `get_bpe_embedding`).
  - If `bpe=False`, the words must exist in the embedding vocabulary directly.

  Args:
      w1 (str): First word or term.
      w2 (str): Second word or term.
      embeddings (gensim KeyedVectors): Word or BPE embedding model.
      bpe (BPE tokenizer, optional): Tokenizer to use for BPE. Defaults to None.

  Returns:
      float or None: Cosine similarity between the two terms, or None if one or both embeddings are unavailable.
  """
  if bpe_tokenizer is not None:
    w1 = get_bpe_embedding(w1, bpe_tokenizer, embeddings)
    w2 = get_bpe_embedding(w2, bpe_tokenizer, embeddings)
  else:
    if w1 in embeddings and w2 in embeddings:
      w1 = embeddings[w1]
      w2 = embeddings[w2]
    else:
      return None
  return cosine_similarity([w1], [w2])[0][0]

def evaluate(data, embeddings, bpe = False):
  """
  Evaluates a word embedding model by computing similarity correlations against
  human-annotated biomedical term pairs.

  - For each term pair in the dataset, the model-based cosine similarity is computed.
  - If `bpe=True`, subword-level embeddings are used via `get_bpe_embedding`.
  - Only pairs where both terms have valid embeddings are included.

  Evaluation metrics:
  - **Spearman correlation**: Measures rank agreement with human scores
  - **Pearson correlation**: Measures linear correlation with human scores
  - **Number Evaluated**: Number of pairs included in the evaluation

  Args:
      data (pd.DataFrame): Evaluation dataset with columns `text_1`, `text_2`, and `mean_score`.
      embeddings (gensim KeyedVectors): Word or subword embedding model.
      bpe (bool, optional): Whether to use BPE-based subword embeddings. Defaults to False.

  Returns:
      inds (List[int]): Indices of rows that were successfully evaluated.
      pd.Series: Spearman correlation, Pearson correlation, and number of evaluated pairs.
  """
  similarities = []
  human_scores = []
  inds = []
  for i, row in data.iterrows():
    sim = get_similarity(row['text_1'].lower(), row['text_2'].lower(), embeddings, bpe)
    if sim is not None:
      inds.append(i)
      similarities.append(sim)
      human_scores.append(row['mean_score'])

  spearman_score = spearmanr(similarities, human_scores)[0]
  pearson_score = pearsonr(similarities, human_scores)[0]
  number_evaluated = len(similarities)

  return inds, pd.Series([spearman_score, pearson_score, number_evaluated])
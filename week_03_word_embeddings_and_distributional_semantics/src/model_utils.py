from gensim.models import Word2Vec

def train_model(vector_size, window, sg, min_count, workers, tokenized_abstracts, epochs, seed = 56): 
    """
    Trains a Word2Vec model using the provided parameters and tokenized abstracts.

    Args:
        vector_size (int): Dimensionality of the word vectors.
        window (int): Maximum distance between the current and predicted word within a sentence.
        sg (int): Skip-gram method if 1, CBOW if 0.
        min_count (int): Ignores all words with total frequency lower than this.
        workers (int): Number of worker threads to train the model.
        tokenized_abstracts (list): List of tokenized abstracts.
        epochs (int): Number of epochs to train the model.

    Returns:
        gensim.models.Word2Vec: Trained Word2Vec model.
    """
    model = Word2Vec(
        vector_size = vector_size, 
        window = window, 
        sg = sg, 
        min_count = min_count, 
        workers = workers, 
        seed = seed
    )

    model.build_vocab(tokenized_abstracts) 
    model.train(tokenized_abstracts, total_examples = model.corpus_count, epochs = epochs)

    return model 
from src.data.chunking import * 
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def preprocess(batch, idx, tokenizer, chunk_len = 512, stride = 412, min_len = 256, max_len = 1024, num_keep = 6, train = True, chunking_strategy = 'middle', embedding_model = None):
    """
    Preprocesses a batch of article-abstract pairs for training or inference.

    Articles are split into chunks using a specified strategy. For dynamic chunking, the most relevant chunks
    are selected based on cosine similarity between chunk and target embeddings.

    Args:
        batch (dict): A batch from a Hugging Face Dataset with 'article' and 'abstract' fields.
        idx (List[int]): List of indices corresponding to the batch samples.
        tokenizer: Hugging Face tokenizer used for encoding input text.
        chunk_len (int): Maximum number of tokens in each chunk.
        stride (int): Number of tokens to move forward when generating overlapping chunks.
        min_len (int): Minimum length of chunk (in tokens) to keep.
        max_len (int): Maximum input length accepted by the model.
        num_keep (int): Number of most relevant chunks to keep during dynamic chunking.
        train (bool): If True, include labels for training. If False, return metadata for evaluation.
        chunking_strategy (str): One of ['middle', 'dynamic'].
                                 'middle' keeps the middle portion of the article;
                                 'dynamic' selects chunks most similar to the abstract or the article centroid.
        embedding_model (Optional): SentenceTransformer model used for computing embeddings in dynamic mode.

    Returns:
        dict: A dictionary with the following keys:
            - 'input_ids': List of tokenized input chunks.
            - 'attention_mask': List of attention masks corresponding to the input chunks.
            - 'labels': (only in train mode) Tokenized abstract sequences as supervision targets.
            - 'article_id': (only in inference mode) Sample indices for tracking.
            - 'reference': (only in inference mode) Ground truth abstracts.
    """
    input_ids = []
    attention_masks = []
    labels = []
    article_ids = []
    references = []

    for i, (article, abstract) in enumerate(zip(batch['article'], batch['abstract'])): 
        if chunking_strategy == 'middle': 
            chunks = chunk_text(article, tokenizer, chunk_len, stride, min_len, return_text = False) 
            chunks = chunks[:num_keep]
        elif chunking_strategy == 'dynamic': 
            if embedding_model is None: 
                raise ValueError('Embedding model must be provided for dynamic chunking.') 

            raw_chunks = chunk_text(article, tokenizer, chunk_len, stride, min_len, return_text = True) 

            if len(raw_chunks) == 0: 
                chunks = chunk_text(article, tokenizer, chunk_len, stride, min_len, return_text = False)
            else: 
                chunk_embeddings = get_embeddings(raw_chunks, embedding_model)
                chunk_embeddings = chunk_embeddings.cpu().numpy()

                if train: 
                    target_embeddings = get_embeddings([abstract], embedding_model).cpu().numpy()
                else: 
                    target_embeddings = np.mean(chunk_embeddings, axis = 0, keepdims = True)

                sims = cosine_similarity(chunk_embeddings, target_embeddings).squeeze()
                inds = np.argsort(sims)[-num_keep:][::-1]

                selected_chunks = [raw_chunks[j] for j in inds]
                if raw_chunks[0] not in selected_chunks: 
                    selected_chunks = [raw_chunks[0]] + selected_chunks

                selected_chunks = selected_chunks[:num_keep]

                chunks = [tokenizer.encode(c, add_special_tokens = False) for c in selected_chunks]

        else: 
            raise ValueError(f'Invalid chunking strategy: {chunking_strategy}')

        abstract_ids = tokenizer.encode(abstract, add_special_tokens = False) 

        for chunk in chunks: 
            if len(chunk) > max_len: 
                chunk = chunk[:max_len]

            attention_mask = [1] * len(chunk) 

            input_ids.append(chunk) 
            attention_masks.append(attention_mask) 

            if train: 
                labels.append(abstract_ids)
            else: 
                article_ids.append(idx[i])
                references.append(abstract)

    if train: 
        return {
            'input_ids': input_ids, 
            'attention_mask': attention_masks, 
            'labels': labels
        }
    else: 
        return {
            'input_ids': input_ids, 
            'attention_mask': attention_masks, 
            'article_id': article_ids, 
            'reference': references
        }
from src.data.chunking import * 

def preprocess(batch, idx, tokenizer, chunk_len = 512, stride = 412, min_len = 256, max_len = 1024, num_keep = 6, train = True, chunking_strategy = 'middle', embedding_model = None):
    """
    Preprocess a batch of text data by chunking and encoding.
    
    Args:
        batch (list): List of text strings to preprocess.
        idx (int): Index of the current batch.
        tokenizer: Tokenizer to use for encoding.
        chunk_len (int): Length of each chunk.
        stride (int): Stride for overlapping chunks.
        min_len (int): Minimum length of chunks to keep.
        return_text (bool): Whether to return text or token IDs.
    
    Returns:
        list: List of processed chunks.
    """
    input_ids = []
    attention_masks = []
    labels = []
    article_ids = []
    references = []

    for i, (article, abstract) in enumerate(zip(batch['article'], batch['abstract'])): 
        if chunking_strategy == 'middle': 
            chunks = chunk_text(article, tokenizer, chunk_len, stride, min_len, return_text = False) 
        elif chunking_strategy == 'dynamic': 
            if embedding_model is None: 
                raise ValueError('Embedding model must be provided for dynamic chunking.') 

            raw_chunks = chunk_text(article, tokenizer, chunk_len, stride, min_len, return_text = True) 

            if len(raw_chunks) == 0: 
                chunks = chunk_text(article, tokenizer, chunk_len, stride, min_len, return_text = False)
            else: 
                chunk_embeddings = get_embeddings(raw_chunks, embedding_model)

                if train: 
                    target_embeddings = get_embeddings([abstract], embedding_model)
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
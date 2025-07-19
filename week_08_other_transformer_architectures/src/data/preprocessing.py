def chunk_text(text, tokenizer, chunk_len = 512, stride = 412, min_len = 256, num_keep = 6): 
    """
    Tokenizes and splits input text into overlapping chunks for processing.

    Args:
        text (str): The input text to be chunked.
        tokenizer: A HuggingFace tokenizer with an `encode` method.
        chunk_len (int): Maximum number of tokens per chunk.
        stride (int): Step size for the sliding window. Controls overlap.
        min_len (int): Minimum number of tokens for a valid chunk.
        num_keep (int): Maximum number of chunks to keep. 
                        If more than `num_keep`, selects middle chunks.

    Returns:
        List[List[int]]: A list of token ID chunks, each a list of integers.
    """
    text_ids = tokenizer.encode(text, add_special_tokens = False)

    chunks = []
    for i in range(0, len(text_ids), stride): 
        chunk = text_ids[i:i + chunk_len]
        if len(chunk) < min_len: 
            break
        chunks.append(chunk)

    if len(chunks) <= num_keep: 
        return chunks 

    third = num_keep // 3  

    start = chunks[:third]

    mid_start = len(chunks) // 2 - (third // 2)
    mid = chunks[mid_start:mid_start + third]

    end = chunks[-third:]

    return start + mid + end

def preprocess(batch, idx, tokenizer, chunk_len = 512, stride = 412, min_len = 256, num_keep = 6, max_len = 1024, train = True): 
    """
    Preprocesses a batch of scientific paper articles and abstracts by splitting articles into 
    overlapping tokenized chunks and pairing them with their corresponding abstract summaries.

    If in training mode, returns tokenized input chunks with abstract labels. 
    If in test/eval mode, returns tokenized inputs along with article IDs and raw references 
    for downstream summary generation and evaluation.

    Args:
        batch (dict): A batch from the Hugging Face dataset with keys 'article' and 'abstract'.
        idx (list): A list of indices corresponding to the current batch (provided via `with_indices=True`).
        tokenizer (PreTrainedTokenizer): The tokenizer used to tokenize input articles and abstracts.
        chunk_len (int, optional): Length of each tokenized chunk. Default is 512.
        stride (int, optional): Overlap between consecutive chunks. Default is 412.
        min_len (int, optional): Minimum token length to keep a chunk. Default is 256.
        num_keep (int, optional): Maximum number of chunks to keep per article. Default is 6.
        max_len (int, optional): Maximum token length for each chunk (applied after chunking). Default is 1024.
        train (bool, optional): Flag indicating whether to return training-format output (with labels) or 
                                evaluation-format output (with article IDs and references). Default is True.

    Returns:
        dict: 
            - If `train=True`: 
                {
                    'input_ids': List[List[int]],
                    'attention_mask': List[List[int]],
                    'labels': List[List[int]]
                }
            - If `train=False`: 
                {
                    'input_ids': List[List[int]],
                    'attention_mask': List[List[int]],
                    'article_id': List[str],
                    'reference': List[str]
                }
    """
    input_ids = []
    attention_masks = []
    labels = []
    article_ids = []
    references = []

    for i, (article, abstract) in enumerate(zip(batch['article'], batch['abstract'])): 
        chunks = chunk_text(article, tokenizer, chunk_len, stride, min_len, num_keep) 
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
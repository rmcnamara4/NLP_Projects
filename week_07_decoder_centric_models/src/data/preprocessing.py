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

    # mid = len(chunks) // 2
    # half = num_keep // 2
    # start = max(0, mid - half) 

    half = num_keep // 2

    return chunks[:half] + chunks[-half:]
    # return chunks[start:start + num_keep]

def train_preprocess(batch, tokenizer, chunk_len, stride = 350, min_len = 256, max_len = 1024): 
    """
    Preprocesses a batch of articles and abstracts into input-target pairs for language model fine-tuning.

    This function:
    - Splits each article into overlapping chunks.
    - Constructs an input prompt of the form: 'Summarize this: <chunk>\\nTL;DR: <summary>'.
    - Applies masking so the model learns to predict only the summary tokens.

    Args:
        batch (Dict[str, List[str]]): A dictionary with two keys:
            - 'article': List of full article texts.
            - 'abstract': List of reference summaries.
        tokenizer (PreTrainedTokenizer): HuggingFace tokenizer used to tokenize and pad inputs.
        chunk_len (int): Maximum length of each chunk (excluding prompt and summary).
        stride (int, optional): Number of tokens to move forward for the next chunk (controls overlap). Default is 350.
        min_len (int, optional): Minimum number of tokens a chunk must have to be used. Default is 256.
        max_len (int, optional): Maximum total length (input + prompt + summary) for each sequence. Default is 1024.

    Returns:
        Dict[str, List[List[int]]]: A dictionary with:
            - 'input_ids': List of token ID sequences.
            - 'attention_mask': List of attention mask sequences.
            - 'labels': List of label sequences with prompt tokens masked as -100.
    """
    input_ids = []
    attention_masks = []
    labels = []

    for article, abstract in zip(batch['article'], batch['abstract']): 
        article_chunks = chunk_text(article, tokenizer, chunk_len, stride, min_len) 
        abstract_ids = tokenizer.encode(abstract, add_special_tokens = False) 
        
        for chunk in article_chunks: 
            prompt_ids = tokenizer.encode('Summarize this: ', add_special_tokens = False) 
            tldr_ids = tokenizer.encode('\nTL;DR: ', add_special_tokens = False)

            input_chunk = prompt_ids + chunk + tldr_ids + abstract_ids 
            if len(input_chunk) > max_len: 
                input_chunk = input_chunk[:max_len]

            attention_mask = [1] * len(input_chunk) 

            label_ids = [-100] * (len(prompt_ids) + len(chunk) + len(tldr_ids)) + abstract_ids
            if len(label_ids) > len(input_chunk): 
                label_ids = label_ids[:len(input_chunk)]
            else: 
                label_ids += [-100] * (len(input_chunk) - len(label_ids))
                        
            input_ids.append(input_chunk) 
            attention_masks.append(attention_mask)
            labels.append(label_ids)

    return {
        'input_ids': input_ids, 
        'attention_mask': attention_masks, 
        'labels': labels
    }

def test_preprocess(batch, idx, tokenizer, chunk_len = 512, stride = 350, min_len = 256, max_len = 1024): 
    """
    Preprocesses a batch of articles for inference-time summarization.

    This function:
    - Splits each article into overlapping chunks using a sliding window.
    - Constructs inputs of the form: 'Summarize this: <chunk>\\nTL;DR:'.
    - Returns chunked inputs along with the original article index (from the dataset)
      and reference summaries for evaluation and postprocessing.

    Args:
        batch (Dict[str, List[str]]): A dictionary containing:
            - 'article': List of full article texts.
            - 'abstract': List of corresponding reference summaries.
        idx (List[int]): The dataset indices corresponding to the current batch.
        tokenizer (PreTrainedTokenizer): HuggingFace tokenizer used to tokenize and encode input text.
        chunk_len (int, optional): Maximum token length of each article chunk (excluding prompt). Default is 512.
        stride (int, optional): Overlap in tokens between adjacent chunks. Default is 350.
        min_len (int, optional): Minimum number of tokens a chunk must have to be included. Default is 256.
        max_len (int, optional): Maximum total token length of the entire input (including prompt and TL;DR). Default is 1024.

    Returns:
        Dict[str, List]: A dictionary with the following keys:
            - 'input_ids': Tokenized input sequences for each chunk.
            - 'attention_mask': Attention masks for the input sequences.
            - 'article_id': The original dataset index for each chunk, used for grouping during postprocessing.
            - 'reference': Repeated reference summaries, one for each chunk.
    """
    input_ids = []
    attention_masks = []
    article_ids = []
    references = []

    for i, (article, abstract) in enumerate(zip(batch['article'], batch['abstract'])): 
        article_chunks = chunk_text(article, tokenizer, chunk_len, stride, min_len)
        prompt_ids = tokenizer.encode('Summarize this: ', add_special_tokens = False) 
        tldr_ids = tokenizer.encode('\nTL;DR: ', add_special_tokens = False)

        for chunk in article_chunks: 
            chunk_input = prompt_ids + chunk + tldr_ids
            if len(chunk_input) > max_len: 
                chunk_input = chunk_input[:max_len]

            attention_mask = [1] * len(chunk_input) 

            input_ids.append(chunk_input) 
            attention_masks.append(attention_mask)
            article_ids.append(idx[i])
            references.append(abstract) 

    return {
        'input_ids': input_ids, 
        'attention_mask': attention_masks, 
        'article_id': article_ids, 
        'reference': references
    }


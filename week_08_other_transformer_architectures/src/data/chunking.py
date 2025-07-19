def chunk_text(text, tokenizer, chunk_len = 512, stride = 412, min_len = 256, return_text = False): 
    """
    Splits input text into overlapping token chunks for processing.

    Args:
        text (str): The input text to be chunked.
        tokenizer: A HuggingFace tokenizer instance used to tokenize and decode text.
        chunk_len (int): Maximum number of tokens in each chunk.
        stride (int): Step size between the starts of consecutive chunks (controls overlap).
        min_len (int): Minimum number of tokens required to keep a chunk.
        return_text (bool): If True, returns decoded text chunks; otherwise, returns token ID chunks.

    Returns:
        List[Union[List[int], str]]: A list of chunks, either as lists of token IDs or decoded text strings,
                                     depending on `return_text`.
    """
    text_ids = tokenizer.encode(text, add_special_tokens = False) 
    
    chunks = []
    for i in range(0, len(text_ids), stride): 
        chunk = text_ids[i:i + chunk_len]
        if len(chunk) < min_len: 
            break 
        chunks.append(chunk) 

    if return_text: 
        return [tokenizer.decode(chunk, skip_special_tokens = True) for chunk in chunks]
    
    return chunks 

def get_embeddings(texts, model): 
    """
    Generates embeddings for a list of texts using a SentenceTransformer-style model.

    Args:
        texts (Union[str, List[str]]): A single string or a list of input text strings to embed.
        model: A SentenceTransformer or similar model with an `.encode()` method.

    Returns:
        torch.Tensor: A tensor containing the embeddings for the input texts.
                      Shape is (num_texts, embedding_dim).
    """
    return model.encode(texts, convert_to_tensor = True) 


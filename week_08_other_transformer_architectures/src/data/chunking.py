def chunk_text(text, tokenizer, chunk_len = 512, stride = 412, min_len = 256, return_text = False): 
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
    return model.encode(texts, convert_to_tensor = True) 


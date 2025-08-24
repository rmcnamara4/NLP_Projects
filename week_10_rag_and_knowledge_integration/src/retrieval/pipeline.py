from src.vectorstore.faiss_store import FaissStore
from typing import Iterable, Dict, List, Any

def load_index(path: str): 
    return FaissStore.load(path)

def load_embedder(embedder_provider: str, path: str): 
    if embedder_provider.lower() in ('hf', 'huggingface'): 
        from src.embeddings.hf import HFEmbeddings
        return HFEmbeddings.load(path)
    elif embedder_provider.lower() == 'bedrock':
        from src.embeddings.bedrock import BedrockEmbeddings
        return BedrockEmbeddings.load(path)
    else: 
        raise ValueError(f'Unknown embedder provider: {embedder_provider}. Provider must be one of "hf" or "bedrock".')
    
def run_search(
    queries: Iterable[Dict[str, Any]], 
    index, 
    embedder, 
    k: int = 5
) -> List[Dict[str, Any]]: 
    results = []
    for q in queries: 
        id = q.get('id', None)
        query_text = q.get('query', None)

        embedded_query = embedder.embed_query(query_text)
        out = index.search(embedded_query, k = k)
        results.append({
            'id': id, 
            'query': query_text, 
            'k': k,
            'results': out, 
            'gold_answer': q.get('gold_answer', None)
        })

    return results 
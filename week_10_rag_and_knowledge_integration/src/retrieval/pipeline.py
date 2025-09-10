from src.vectorstore.faiss_store import FaissStore
from typing import Iterable, Dict, List, Any

def load_index(path: str): 
    """
    Load a FAISS index from a given path.

    Args:
        path: Filesystem or S3 path where the FAISS index is stored.

    Returns:
        FaissStore: The loaded FAISS index object.
    """
    return FaissStore.load(path)

def load_embedder(embedder_provider: str, path: str): 
    """
    Load an embedding model from a given provider and path.

    Args:
        embedder_provider: The provider of the embeddings. 
            Supported values are "hf" / "huggingface" and "bedrock".
        path: Filesystem or S3 path to the saved embedder configuration.

    Returns:
        Embeddings: An instance of the loaded embedding model.

    Raises:
        ValueError: If the provider is not recognized.
    """
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
    """
    Run a semantic search over an index for a batch of queries.

    Args:
        queries: Iterable of query dictionaries, each containing:
            - 'id': Optional unique identifier for the query.
            - 'query': The query string to embed and search.
            - 'gold_answer': Optional ground-truth answer for evaluation.
        index: Search index object with a `.search(embedding, k)` method (e.g., FAISS).
        embedder: Embedding model with an `embed_query(query: str)` method.
        k: Number of nearest neighbors to return for each query. Default is 5.

    Returns:
        List[Dict[str, Any]]: Search results for each query, with keys:
            - 'id': Query ID (if provided).
            - 'query': Original query text.
            - 'k': Number of results requested.
            - 'results': Top-k retrieved items from the index.
            - 'gold_answer': Original ground-truth answer (if provided).
    """
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
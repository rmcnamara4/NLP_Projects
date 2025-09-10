import os, json 
from typing import List, Optional, Dict, Any 
import numpy as np 
import faiss

from src.vectorstore.base import VectorStore

def _l2_normalize(vectors: np.ndarray) -> np.ndarray: 
    """
    Apply L2 normalization to a batch of vectors.

    Each row vector is divided by its L2 norm so that all output vectors
    have unit length. A small epsilon (1e-12) is added to the denominator
    for numerical stability.

    Args:
        vectors (np.ndarray): 2D array of shape (n, d), where each row is a vector.

    Returns:
        np.ndarray: L2-normalized array of the same shape as input.
    """
    norms = np.linalg.norm(vectors, axis = 1, keepdims = True) + 1e-12
    return vectors / norms

class FaissStore(VectorStore):
    """
    A FAISS-backed vector store for similarity search.

    Supports both flat (IndexFlatIP/L2) and IVF (IndexIVFFlat) indices,
    with optional L2 normalization for cosine similarity via inner product.
    Stores metadata alongside embeddings for later retrieval.


    Args:
        dim (int): Embedding dimension.
        metric (str): Distance metric to use: 
            - 'ip' for inner product (cosine if normalize=True).
            - 'l2' for Euclidean distance.
        normalize (bool): Whether to L2-normalize vectors before adding/searching.
    """
    def __init__(self, dim: int, metric: str = 'ip', normalize: bool = True): 
        self.dim = dim 
        self.metric = metric 
        self.normalize = normalize

        self.index = faiss.IndexFlatIP(dim) if metric == 'ip' else faiss.IndexFlatL2(dim)
        self._metadatas: List[Dict[str, Any]] = []

    @classmethod 
    def ivf(cls, dim: int, nlist: int = 1024, metric: str = 'ip', normalize: bool = True) -> 'FaissStore': 
        """
        Create a FAISS IVF (inverted file) index for approximate nearest neighbor search.

        Args:
            dim (int): Embedding dimension.
            nlist (int): Number of clusters (inverted lists) for IVF index.
            metric (str): 'ip' (inner product) or 'l2' (Euclidean distance).
            normalize (bool): Whether to L2-normalize vectors.

        Returns:
            FaissStore: An instance with IVF index backend.
        """
        quant = faiss.IndexFlatIP(dim) if metric == 'ip' else faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFFlat(quant, dim, nlist, faiss.METRIC_INNER_PRODUCT if metric == 'ip' else faiss.METRIC_L2) 
        obj = cls(dim, metric, normalize) 
        obj.index = index 
        return obj 

    def add(self, embeddings: np.ndarray, metadatas: List[Dict[str, Any]]) -> None: 
        """
        Add vectors and their associated metadata to the index.

        Args:
            embeddings (np.ndarray): Array of shape (n, d) containing vectors.
            metadatas (List[Dict[str, Any]]): List of metadata dicts, one per vector.

        Notes:
            If using IVF, the index is trained automatically on first add.
        """
        vecs = embeddings.astype(np.float32) 
        if self.normalize: vecs = _l2_normalize(vecs)
        if isinstance(self.index, faiss.IndexIVF) and not self.index.is_trained:  
            self.index.train(vecs) 
        self.index.add(vecs) 
        self._metadatas.extend(metadatas) 

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]: 
        """
        Perform similarity search on the index.

        Args:
            query_embedding (np.ndarray): Query vector of shape (d,).
            k (int): Number of nearest neighbors to return.

        Returns:
            List[Dict[str, Any]]: Metadata dicts for top-k matches, 
            each augmented with a 'score' field.
        """
        q_vec = query_embedding.astype(np.float32).reshape(1, -1) 
        if self.normalize: q_vec = _l2_normalize(q_vec)
        D, I = self.index.search(q_vec, k) 
        out = []
        for score, idx in zip(D[0], I[0]): 
            if idx < 0: continue 
            rec = dict(self._metadatas[idx])
            rec['score'] = float(score) 
            out.append(rec) 

        return out 
    
    def save(self, path: str) -> None: 
        """
        Save index and metadata to disk.

        Args:
            path (str): Directory to save files. Creates if not exists.

        Notes:
            - Saves index as `index.faiss`
            - Saves metadata as `metadatas.jsonl`
            - Saves config as `store.json`
        """
        os.makedirs(path, exist_ok = True) 
        faiss.write_index(self.index, os.path.join(path, 'index.faiss'))
        with open(os.path.join(path, 'metadatas.jsonl'), 'w') as f:
            for m in self._metadatas:
                f.write(json.dumps(m) + '\n') 
        with open(os.path.join(path, 'store.json'), 'w') as f:
            json.dump({
                'dim': self.dim, 
                'metric': self.metric, 
                'normalize': self.normalize
            }, f)

    @classmethod
    def load(cls, path: str) -> 'FaissStore': 
        """
        Load a FAISS store from disk.

        Args:
            path (str): Directory containing saved index, metadata, and config.

        Returns:
            FaissStore: Restored vector store instance with loaded index and metadata.
        """
        with open(os.path.join(path, 'store.json')) as f: 
            cfg = json.load(f) 

        obj = cls(cfg['dim'], cfg['metric'], cfg['normalize'])
        obj.index = faiss.read_index(os.path.join(path, 'index.faiss'))
        with open(os.path.join(path, 'metadatas.jsonl')) as f: 
            obj._metadatas = [json.loads(line) for line in f]

        return obj



    


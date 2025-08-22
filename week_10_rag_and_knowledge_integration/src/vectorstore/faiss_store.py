import os, json 
from typing import List, Optional, Dict, Any 
import numpy as np 
import faiss

from src.vectorstore.base import VectorStore

def _l2_normalize(vectors: np.ndarray) -> np.ndarray: 
    norms = np.linalg.norm(vectors, axis = 1, keepdims = True) + 1e-12
    return vectors / norms

class FaissStore(VectorStore):
    def __init__(self, dim: int, metric: str = 'ip', normalize: bool = True): 
        self.dim = dim 
        self.metric = metric 
        self.normalize = normalize

        self.index = faiss.IndexFlatIP(dim) if metric == 'ip' else faiss.IndexFlatL2(dim)
        self._metadatas: List[Dict[str, Any]] = []

    # @classmethod 

    def add(self, embeddings: np.ndarray, metadatas: List[Dict[str, Any]]) -> None: 
        vecs = embeddings.astype(np.float32) 
        if self.normalize: vecs = _l2_normalize(vecs)
        if isinstance(self.index, faiss.IndexIVF) and not self.index.is_trained:  
            self.index.train(vecs) 
        self.index.add(vecs) 
        self._metadatas.extend(metadatas) 

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]: 
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
        with open(os.path.join(path, 'store.json')) as f: 
            cfg = json.load(f) 

        obs = cls(cfg['dim'], cfg['metric'], cfg['normalize'])
        obj.index = faiss.read_index(os.path.join(path, 'index.faiss'))
        with open(os.path.join(path, 'metadatas.json')) as f: 
            obj._metadatas = [json.loads(line) for line in f]

        return obj



    


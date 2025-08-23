from typing import List
import numpy as np 
import os
import json
from sentence_transformers import SentenceTransformer
from src.embeddings.base import Embeddings 

class HFEmbeddings(Embeddings):
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2', device: str | None = None):
        self.model_name = model_name
        self.device = device if device is not None else 'cpu'
        self.model = SentenceTransformer(model_name, device = device)

        _d = self.model.encode(['_']).astype(np.float32) 
        self._dim = _d.shape[1]

    def embed_text(self, texts: List[str], batch_size: int = 32) -> np.ndarray: 
        embeddings = self.model.encode(
            texts, 
            batch_size = batch_size, 
            convert_to_numpy = True, 
            normalize_embeddings = False
        )

        return embeddings.astype('float32')
    
    def embed_query(self, query: str) -> np.ndarray:
        embedding = self.model.encode(
            [query], 
            convert_to_numpy = True, 
            normalize_embeddings = False
        )[0]
    
        return embedding.astype('float32')
    
    @property
    def dim(self) -> int: 
        return int(self._dim)
    
    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok = True) 
        with open(os.path.join(path, 'embedder.json'), 'w') as f: 
            json.dump({
                'embedder_provider': 'hf',
                'model_name': self.model_name, 
                'device': self.device
            }, f)

    @classmethod
    def load(cls, path: str) -> 'Embeddings':
        with open(os.path.join(path, 'embedder.json')) as f: 
            info = json.load(f) 
        return cls(
            model_name = info['model_name'], 
            device = info.get('device', None)
        )

    
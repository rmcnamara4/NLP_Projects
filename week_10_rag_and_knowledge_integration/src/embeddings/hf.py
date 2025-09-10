from typing import List
import numpy as np 
import os
import json
from sentence_transformers import SentenceTransformer
from src.embeddings.base import Embeddings 

class HFEmbeddings(Embeddings):
    """
    Embeddings implementation using Hugging Face's SentenceTransformers.

    Args:
        model_name (str): Pretrained model to load. 
            Defaults to 'sentence-transformers/all-MiniLM-L6-v2'.
        device (str | None): Device to run on (e.g., 'cpu', 'cuda'). 
            If 'cpu', falls back to default SentenceTransformer behavior.

    This class wraps SentenceTransformers to generate embeddings for both
    documents and queries, with support for saving and reloading configuration.
    """
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2', device: str | None = None):
        self.model_name = model_name
        self.device = device if device != 'cpu' else None
        self.model = SentenceTransformer(model_name, device = self.device)

        _d = self.model.encode(['_']).astype(np.float32) 
        self._dim = _d.shape[1]

    def embed_text(self, texts: List[str], batch_size: int = 32) -> np.ndarray: 
        """
        Generate embeddings for a list of input texts.

        Args:
            texts (List[str]): Input documents to embed.
            batch_size (int): Batch size for encoding. Defaults to 32.

        Returns:
            np.ndarray: Embeddings with shape (len(texts), dim).
        """
        embeddings = self.model.encode(
            texts, 
            batch_size = batch_size, 
            convert_to_numpy = True, 
            normalize_embeddings = False
        )

        return embeddings.astype('float32')
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate an embedding for a single query string.

        Args:
            query (str): Input query text.

        Returns:
            np.ndarray: Embedding vector with shape (dim,).
        """
        embedding = self.model.encode(
            [query], 
            convert_to_numpy = True, 
            normalize_embeddings = False
        )[0]
    
        return embedding.astype('float32')
    
    @property
    def dim(self) -> int: 
        """
        Embedding dimensionality.

        Returns:
            int: Size of embedding vectors.
        """
        return int(self._dim)
    
    def save(self, path: str) -> None:
        """
        Save embedder configuration to disk.

        Args:
            path (str): Directory path to save configuration (embedder.json).
        """
        os.makedirs(path, exist_ok = True) 
        with open(os.path.join(path, 'embedder.json'), 'w') as f: 
            json.dump({
                'embedder_provider': 'hf',
                'model_name': self.model_name, 
                'device': self.device
            }, f)

    @classmethod
    def load(cls, path: str) -> 'Embeddings':
        """
        Load embedder configuration from disk.

        Args:
            path (str): Directory containing embedder.json.

        Returns:
            HFEmbeddings: Restored embedder instance.
        """
        with open(os.path.join(path, 'embedder.json')) as f: 
            info = json.load(f) 
        return cls(
            model_name = info['model_name'], 
            device = info.get('device', None)
        )

    
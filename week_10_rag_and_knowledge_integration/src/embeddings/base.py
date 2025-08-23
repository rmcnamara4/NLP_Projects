from abc import ABC, abstractmethod
from typing import List
import numpy as np

class Embeddings(ABC): 
    @abstractmethod 
    def embed_text(self, texts: List[str], batch_size: int = 32) -> np.ndarray: 
        pass

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def dim(self) -> int: 
        pass

    @abstractmethod
    def save(self, path: str) -> None: 
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> 'Embeddings':
        pass 
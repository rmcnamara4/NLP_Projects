from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np 

class VectorStore(ABC): 
    @abstractmethod
    def add(self, embeddings: np.ndarray, metadatas: List[Dict[str, Any]]) -> None: 
        pass

    @abstractmethod
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]: 
        pass 

    @abstractmethod
    def save(self, path: str) -> None: 
        pass 

    @classmethod 
    @abstractmethod
    def load(cls, path: str) -> 'VectorStore': 
        pass
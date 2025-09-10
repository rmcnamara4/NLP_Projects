from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np 

class VectorStore(ABC): 
    """
    Abstract base class for vector store implementations.

    A vector store manages dense vector embeddings and their associated metadata,
    supporting insertion, similarity search, persistence, and reloading.
    Subclasses must implement all abstract methods.
    """

    @abstractmethod
    def add(self, embeddings: np.ndarray, metadatas: List[Dict[str, Any]]) -> None: 
        """
        Add a batch of embeddings and associated metadata to the store.

        Args:
            embeddings (np.ndarray): 2D array of shape (n, d) containing the vectors to store.
            metadatas (List[Dict[str, Any]]): Metadata dictionaries corresponding to each vector.
        """
        pass

    @abstractmethod
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]: 
        """
        Perform a similarity search against the stored vectors.

        Args:
            query_embedding (np.ndarray): 1D array representing the query vector.
            k (int, optional): Number of nearest neighbors to return. Defaults to 5.

        Returns:
            List[Dict[str, Any]]: Top-k metadata records for the closest embeddings.
        """
        pass 

    @abstractmethod
    def save(self, path: str) -> None: 
        """
        Persist the vector store to disk.

        Args:
            path (str): Filesystem path to save the serialized store.
        """
        pass 

    @classmethod 
    @abstractmethod
    def load(cls, path: str) -> 'VectorStore': 
        """
        Load a vector store from a saved file.

        Args:
            path (str): Filesystem path to the saved vector store.

        Returns:
            VectorStore: An instance of the loaded vector store.
        """
        pass
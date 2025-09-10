from abc import ABC, abstractmethod
from typing import List
import numpy as np

class Embeddings(ABC): 
    """
    Abstract base class for embedding models.

    Provides a consistent interface for embedding text/queries into 
    dense vectors, along with methods to save/load model state.
    """

    @abstractmethod 
    def embed_text(self, texts: List[str], batch_size: int = 32) -> np.ndarray: 
        """
        Embed a batch of text strings.

        Args:
            texts (List[str]): Input text strings.
            batch_size (int): Batch size for processing. Defaults to 32.

        Returns:
            np.ndarray: 2D array of shape (len(texts), dim).
        """
        pass

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string.

        Args:
            query (str): Query text.

        Returns:
            np.ndarray: 1D array of shape (dim,).
        """
        pass

    @property
    @abstractmethod
    def dim(self) -> int: 
        """
        Embedding dimensionality.

        Returns:
            int: Number of vector dimensions.
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None: 
        """
        Save the embedding model.

        Args:
            path (str): File path for saving.
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> 'Embeddings':
        """
        Load a saved embedding model.

        Args:
            path (str): File path to load from.

        Returns:
            Embeddings: Loaded embedding model instance.
        """
        pass 
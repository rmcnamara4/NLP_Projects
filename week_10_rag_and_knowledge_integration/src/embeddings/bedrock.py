from typing import List
import numpy as np 

import os
import boto3
import json 

from src.embeddings.base import Embeddings

from dotenv import load_dotenv
load_dotenv()

def _bedrock_client(): 
    """
    Create and return a Bedrock runtime client using boto3.

    Uses the AWS region specified in the `AWS_DEFAULT_REGION` environment 
    variable, falling back to "us-east-1" if not set.

    Returns:
        boto3.client: A boto3 client for the Bedrock runtime service.
    """
    return boto3.client(
        'bedrock-runtime', 
        region_name = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
    )

class BedrockEmbeddings(Embeddings): 
    """
    Embeddings implementation using AWS Bedrock embedding models (Titan, Cohere).

    Args:
        model_id (Optional[str]): Bedrock model ID. Defaults to env var 
            `BEDROCK_EMBEDDING_MODEL_ID` or 'amazon.titan-embed-text-v2:0'.
        dim (Optional[int]): Embedding dimensionality. Inferred if not provided.
        client: boto3 Bedrock client. Defaults to a new client via `_bedrock_client()`.

    Raises:
        ValueError: If model family is not Titan or Cohere.

    This class supports both text embeddings (documents) and query embeddings, 
    with automatic dimensionality inference, model invocation, saving, and loading.
    """
    def __init__(
            self, 
            model_id: Optional[str] = None, 
            dim: Optional[int] = None, 
            client = None 
    ): 
        self.model_id = model_id or os.getenv('BEDROCK_EMBEDDING_MODEL_ID', 'amazon.titan-embed-text-v2:0')
        self._client = client or _bedrock_client()

        mi = self.model_id.lower()
        if ('titan' not in mi) and ('cohere' not in mi): 
            raise ValueError(f'Unsupported Bedrock embedding model: {self.model_id}. Supported families: Amazon Titan, Cohere.')

        self._dim = int(dim) if dim is not None else self._infer_dim()

    def embed_text(self, texts: List[str], batch_size: int = 32) -> np.ndarray: 
        """
        Generate embeddings for a list of input texts.

        Args:
            texts (List[str]): Input documents to embed.
            batch_size (int): Batch size for Cohere models. Ignored for Titan.

        Returns:
            np.ndarray: Array of embeddings with shape (len(texts), dim).
        """
        mi = self.model_id.lower()
        out = []

        if 'titan' in mi: 
            for t in texts: 
                payload = {'inputText': t}
                result = self._invoke(payload) 
                out.append(result['embedding'])
        else: 
            for i in range(0, len(texts), batch_size): 
                batch = texts[i:i + batch_size]
                payload = {
                    'texts': batch, 
                    'input_type': 'search_document'
                }
                result = self._invoke(payload) 
                out.extend(result['embeddings'])

        return np.asarray(out, dtype = np.float32)

    def embed_query(self, query: str) -> np.ndarray: 
        """
        Generate an embedding for a single query string.

        Args:
            query (str): Query text.

        Returns:
            np.ndarray: Embedding vector with shape (dim,).
        """
        mi = self.model_id.lower()
        if 'titan' in mi:
            payload = {'inputText': query}
            result = self._invoke(payload) 
            embedding = result['embedding'][0]
        else: 
            payload = {
                'texts': [query], 
                'input_type': 'search_query'
            }
            result = self._invoke(payload) 
            embedding = result['embeddings'][0]

        return np.asarray(embedding, dtype = np.float32)

    @property
    def dim(self) -> int: 
        """
        Embedding dimensionality.

        Returns:
            int: Size of embedding vectors.
        """
        return int(self._dim)
    
    def _infer_dim(self) -> int: 
        """
        Infer embedding dimensionality by calling the model on a dummy input.

        Returns:
            int: Inferred dimensionality.
        """
        mi = self.model_id.lower()
        dummy = '_'

        if 'titan' in mi:
            payload = {'inputText': dummy}
            result = self._invoke(payload) 
            return len(result['embedding'])
        
        payload = {
            'texts': [dummy], 
            'input_type': 'search_document'
        }
        result = self._invoke(payload) 
        return len(result['embeddings'][0])
    
    def _invoke(self, payload: dict) -> dict: 
        """
        Call the Bedrock model with a given payload.

        Args:
            payload (dict): JSON payload for the model.

        Returns:
            dict: Parsed JSON response from Bedrock.
        """
        resp = self._client.invoke_model(
            modelId = self.model_id, 
            body = json.dumps(payload).encode('utf-8'), 
            contentType = 'application/json', 
            accept = 'application/json'
        )
        body = resp['body'].read()
        return json.loads(body)
    
    def save(self, path: str) -> None: 
        """
        Save embedder configuration to disk.

        Args:
            path (str): Directory to save configuration (embedder.json).
        """
        os.makedirs(path, exist_ok = True) 
        with open(os.path.join(path, 'embedder.json'), 'w') as f: 
            json.dump({
                'embedder_provider': 'bedrock',
                'model_id': self.model_id, 
                'dim': self.dim
            }, f)

    @classmethod
    def load(cls, path: str) -> 'Embeddings':
        """
        Load embedder configuration from disk.

        Args:
            path (str): Directory containing embedder.json.

        Returns:
            BedrockEmbeddings: Restored embedder instance.
        """
        with open(os.path.join(path, 'embedder.json')) as f: 
            cfg = json.load(f) 
        return cls(model_id = cfg['model_id'], dim = cfg['dim'])
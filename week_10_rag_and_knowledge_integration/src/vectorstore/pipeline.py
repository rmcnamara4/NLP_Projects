from typing import Dict, Iterable

def create_index(records: Iterable[Dict], embedder, store, batch_size: int = 32) -> None: 
    """
    Build and populate a vector index from text records.

    Iterates over records in batches, embeds the text fields using the provided
    embedder, and adds the resulting vectors with their metadata to the store.

    Args:
        records (Iterable[Dict]): Collection of records, each containing a 'text'
            field and optional metadata fields.
        embedder: An object implementing `embed_text`, which converts a list of
            strings into embeddings (np.ndarray).
        store: A vector store implementing `add`, which ingests embeddings with metadata.
        batch_size (int): Number of texts to embed per batch. Defaults to 32.

    Returns:
        None

    Notes:
        - Metadata is all fields of each record except 'text'.
        - Any leftover texts smaller than batch_size are embedded at the end.
    """
    texts, metadatas = [], []
    for r in records: 
        texts.append(r['text'])
        metadatas.append(
            {k: r.get(k, None) for k in r.keys() if k != 'text'}
        )
        if len(texts) >= batch_size: 
            embeddings = embedder.embed_text(texts, batch_size = batch_size) 
            store.add(embeddings, metadatas)
            texts, metadatas = [], []

    if texts: 
        embeddings = embedder.embed_text(texts, batch_size = batch_size) 
        store.add(embeddings, metadatas)
from typing import Dict, Iterable

def create_index(records: Iterable[Dict], embedder, store, batch_size: int = 32) -> None: 
    for r in records: 
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
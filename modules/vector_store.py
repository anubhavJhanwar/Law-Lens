import os
os.environ["TRANSFORMERS_NO_TF"] = "1"  # Disable TensorFlow before importing

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load model once globally
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def create_faiss_index(text, chunk_size=500, overlap=100):
    """Split text into chunks and build FAISS index."""
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])

    embeddings = embedder.encode(chunks, show_progress_bar=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))

    return index, chunks

def search_similar_chunks(query, index, chunks, top_k=3):
    """Return top_k most similar chunks for a query."""
    query_vec = embedder.encode([query])
    D, I = index.search(np.array(query_vec).astype("float32"), top_k)
    results = [chunks[i] for i in I[0]]
    return results

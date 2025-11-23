import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def create_embeddings(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks, show_progress_bar=True)
    return embeddings

def create_faiss_index(embeddings):
    dimension = embeddings[0].shape[0]  # 384 for MiniLM
    index = faiss.IndexFlatL2(dimension)  # L2 distance
    index.add(np.array(embeddings))       # add all vectors
    return index
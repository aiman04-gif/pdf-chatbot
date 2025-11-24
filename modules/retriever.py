import numpy as np

def retrieve_relevant_chunks(question, faiss_index, chunks, model, top_k=3):
    """
    Retrieve the top-k most relevant chunks for a given question.
    
    Parameters:
        question (str): The user's question.
        faiss_index (faiss.IndexFlatL2): The FAISS vector store.
        chunks (list of str): Original text chunks.
        model (SentenceTransformer): The embedding model.
        top_k (int): Number of chunks to retrieve.
        
    Returns:
        List[str]: Top-k relevant text chunks.
    """
    question_embedding = model.encode([question])
    question_vector = np.array(question_embedding).astype('float32')

    distances, indices = faiss_index.search(question_vector, top_k)

    relevant_chunks = [chunks[i] for i in indices[0]]

    return relevant_chunks
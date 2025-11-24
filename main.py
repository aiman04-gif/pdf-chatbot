from sentence_transformers import SentenceTransformer
from modules.pdf_loader import extract_text_from_pdf
from modules.text_splitter import split_text_into_chunks
from modules.embeddings import create_embeddings, create_faiss_index
from modules.retriever import retrieve_relevant_chunks
from modules.answer_generator import generate_answer


text = extract_text_from_pdf("data/PPIT - A2 - l227830-l226198-l227955.pdf")
chunks = split_text_into_chunks(text)

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = create_embeddings(chunks)
faiss_index = create_faiss_index(embeddings)

question = "What is the main topic of this pdf?"
top_chunks = retrieve_relevant_chunks(question, faiss_index, chunks, model, top_k=3)

answer = generate_answer(question, top_chunks)
print("\n--- Answer ---\n")
print(answer)

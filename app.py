import streamlit as st
from sentence_transformers import SentenceTransformer
from modules.pdf_loader import extract_text_from_pdf
from modules.text_splitter import split_text_into_chunks
from modules.embeddings import create_embeddings, create_faiss_index
from modules.retriever import retrieve_relevant_chunks
from modules.answer_generator import generate_answer
from modules.hyde_generator import generate_hypothetical_answer
import time

st.markdown(
    """
    <style>
    body {background-color: black; color: black; font-family: Arial, sans-serif;}
    .chat-container {max-height: 500px; overflow-y: auto; padding: 10px; border-radius: 10px;}
    .user-bubble {background-color: #9DC183; color: black; padding: 10px; border-radius: 10px; max-width: 80%; float: right; clear: both; margin-bottom: 5px;}
    .assistant-bubble {background-color: #A9A9A9; color: black; padding: 10px; border-radius: 10px; max-width: 80%; float: left; clear: both; margin-bottom: 5px;}
    header, footer {visibility: hidden;}
    .chat-container::-webkit-scrollbar {width: 6px;}
    .chat-container::-webkit-scrollbar-thumb {background-color: #555; border-radius: 3px;}
    </style>
    """,
    unsafe_allow_html=True
)
def display_message(role, content):
    if role == "user":
        st.markdown(f'<div class="user-bubble">{content}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="assistant-bubble">{content}</div>', unsafe_allow_html=True)

if "embed_model" not in st.session_state:
    st.session_state["embed_model"] = SentenceTransformer('all-MiniLM-L6-v2')
model = st.session_state["embed_model"]

st.title("PDF Chatbot (RAG System)")
use_hyde = st.toggle("Use HyDE Retrieval", value=False)
st.subheader("Upload PDF")
uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file and st.session_state.get("processed_file_name") != uploaded_file.name:
    st.session_state.pop("processed_file_name", None)
    st.session_state.pop("pdf_text", None)
    st.session_state.pop("chunks", None)
    st.session_state.pop("embeddings", None)
    st.session_state.pop("faiss_index", None)
    
if uploaded_file and "processed_file_name" not in st.session_state:
    with st.spinner("Processing PDF..."):
        start_time = time.time()

        pdf_text = extract_text_from_pdf(uploaded_file)
        st.session_state["pdf_text"] = pdf_text

        chunks = split_text_into_chunks(pdf_text)
        st.session_state["chunks"] = chunks

        embeddings = create_embeddings(chunks)
        st.session_state["embeddings"] = embeddings

        faiss_index = create_faiss_index(embeddings)
        st.session_state["faiss_index"] = faiss_index

        # Store file name to avoid reprocessing on every rerun
        st.session_state["processed_file_name"] = uploaded_file.name

        st.success(f"PDF processed! ({round(time.time() - start_time, 2)}s)")


if "messages" not in st.session_state:
    st.session_state["messages"] = []

st.subheader("Chat with your PDF")

chat_container = st.container()
chat_container.markdown('<div class="chat-container">', unsafe_allow_html=True)
for message in st.session_state["messages"]:
    display_message(message["role"], message["content"])
chat_container.markdown('</div>', unsafe_allow_html=True)

user_input = st.chat_input("Ask a question...")

if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    display_message("user", user_input)
    if "faiss_index" in st.session_state and "chunks" in st.session_state:
        with st.chat_message("assistant"):
            with st.spinner("Retrieving answer..."):
                start_time = time.time()
                # If HyDE toggle ON → generate hypothetical answer first
                if use_hyde:
                    hyde_text = generate_hypothetical_answer(user_input)
                    query_for_retrieval = hyde_text
                else:
                    query_for_retrieval = user_input

                retrieved_chunks = retrieve_relevant_chunks(
                    query_for_retrieval,
                    st.session_state["faiss_index"],
                    st.session_state["chunks"],
                    model,
                    top_k=3
                )

                answer = generate_answer(user_input, retrieved_chunks)
                st.markdown(answer)
                st.session_state["messages"].append({"role": "assistant", "content": answer})
                st.session_state["debug"] = {
                    "retrieved_chunks": retrieved_chunks,
                    "time_taken": round(time.time() - start_time, 2),
                    "used_hyde": use_hyde,
                    "hyde_text": hyde_text if use_hyde else None
                }
    else:
        display_message("assistant", "Please upload and process a PDF first.")

with st.expander("Debug Info"):
    if "debug" in st.session_state:
        st.write("Retrieved Chunks:")
        st.write(st.session_state["debug"]["retrieved_chunks"])
        st.write("Time Taken:")
        st.write(f"{st.session_state['debug']['time_taken']} seconds")
        st.write("Used HyDE:", st.session_state["debug"]["used_hyde"])
        if st.session_state["debug"]["used_hyde"]:
            st.write("HyDE Query:")
            st.write(st.session_state["debug"]["hyde_text"])

    else:
        st.write("Debug info will appear here after first query.")

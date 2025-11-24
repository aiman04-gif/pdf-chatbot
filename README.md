## STEP 01 - Setup
1. Setting up environment\
`conda create --name chatbot`

2. Adding necessary installs\
`pip install langchain`\
`pip install sentence-transformers`\
`pip install faiss-cpu`\
`pip install PyPDF2`\

## Step 02 - PDF Loading + Text Extraction
PDF loaded using PyPDF2 the text extracted and stored.
**[see modules/pdf_loader.py]**

## Step 03 - Chunking
The loaded text is then broken down into chunks two imp terms
- chunk_size: max characters in a chunk
- chunk_overlap: characters repeated between chunks
**Why Split into chunks?**
- LLMs cannot take entire PDFs at once 
- Embeddings work better on smalller pieces
- Question corresponds to only specific part of the document.

**[see modules/text_splitter]**

## Step 04 - Embeddings and FAISS Vector Store
For embeddings, sentence-transformers/all-MiniLM-L6-v2 is used. Each chunk will be a vector of 384 dimensions for this model. When user asks a question, the question will also be converted to a vector and compared to the stored vector chunks. 
So *embeddings are the bridge between text and semantic search.*
FAISS is a library for fast vector similarity search

**[see modules/embeddings.py]**

## Step 05 - Querying and Chatbot Logic
Main steps
1. Embed the user question
2. Retrieve top-k similar chunks from FAISS
3. Pass them to an LLM (transformers model) to generate the answer

**[see modules/retriever.py]**

## Step 06 - Answer Generation
In this step all the information *retrieved* is passes to an LLM to *augment* a response. (RAG essentially)
As the requirement stated to external api, a local Hugging Face model is used. Based on resources, **"google/flan-t5-small"** is used.

**[see modules/answer_generator.py]**


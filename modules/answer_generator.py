from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()  # Load .env file

api_key = os.getenv("GROQ_API_KEY")

client = Groq(api_key=api_key)

def generate_answer(question, context_chunks, model_name="llama-3.1-8b-instant"):
    """
    Generate a concise answer using Groq LLM + retrieved context.
    """

    context = " ".join(context_chunks)

    prompt = f"""
        You are an academic assistant answering questions from a PDF.

        Guidelines:
        - Answer using the context.
        - Provide short, precise answers.
        - If multiple points exist, summarize them.
        - If the answer is not explicitly in the context, respond: "Not mentioned in the provided document."

        Context:
        {context}

        Question: {question}

        Answer:


"""

    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=300,
    )

    return completion.choices[0].message.content

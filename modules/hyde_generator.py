from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

api_key=os.getenv("GROQ_API_KEY")
client=Groq(api_key=api_key)

def generate_hypothetical_answer(question, model_name="llama-3.1-8b-instant"):

    prompt = f"""
    You are generating a hypothetical answer to help with document retrieval.

    IMPORTANT:
    - Give a short, plausible answer to the question.
    - It does NOT need to be correct.
    - It must be a coherent guess that captures likely keywords.
    - Do NOT say you are guessing. Just provide the hypothetical answer.

    Question: {question}

    Hypothetical Answer:
    """

    completion = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=120,
    )

    return completion.choices[0].message.content.strip()
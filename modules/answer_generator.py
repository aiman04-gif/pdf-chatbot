from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

def generate_answer(question, context_chunks, model_name="google/flan-t5-small"):
    """
    Generate an answer to the question using the context chunks.
    """
    context = " ".join(context_chunks)

   
    prompt = f"Answer the question based on the context below:\n\nContext: {context}\n\nQuestion: {question}\nAnswer:"

    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    
    nlp = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

    
    result = nlp(prompt, max_length=200, do_sample=False)

    return result[0]['generated_text']

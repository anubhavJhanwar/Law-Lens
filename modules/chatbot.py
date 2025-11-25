import os
from groq import Groq
from modules.vector_store import search_similar_chunks

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def answer_query_with_context(query, index, chunks):
    """Use FAISS + Llama to answer based on document content."""
    context_chunks = search_similar_chunks(query, index, chunks)
    context = "\n\n".join(context_chunks)

    prompt = f"""
You are a legal assistant. Use the following context to answer the user question.
If the answer is not in the document, say "The document does not contain that information."

Context:
{context}

Question: {query}
Answer (in 2-3 concise sentences):
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.strip()

import os
import numpy as np
import faiss
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ----------------------------
# 1. Sample Documents
# ----------------------------
docs = [
    "RAG stands for Retrieval Augmented Generation.",
    "It improves LLM responses by retrieving external knowledge.",
    "FAISS is a vector database used for similarity search.",
    "Embeddings convert text into numerical vectors.",
    "Large language models generate answers based on context."
]

# ----------------------------
# 2. Chunking (simple version)
# ----------------------------
def chunk_text(text):
    return text.split(". ")

chunks = []
for doc in docs:
    chunks.extend(chunk_text(doc))

chunks = [c.strip() for c in chunks if c.strip()]

# ----------------------------
# 3. Embedding function
# ----------------------------
def embed(texts):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return np.array([d.embedding for d in response.data], dtype="float32")

# ----------------------------
# 4. Build Vector Index (FAISS)
# ----------------------------
embeddings = embed(chunks)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# ----------------------------
# 5. Retrieval function
# ----------------------------
def retrieve(query, k=3):
    query_vec = embed([query])
    distances, indices = index.search(query_vec, k)
    return [chunks[i] for i in indices[0]]

# ----------------------------
# 6. Generate answer using LLM
# ----------------------------
def generate_answer(query):
    context_chunks = retrieve(query)

    context = "\n".join(context_chunks)

    prompt = f"""
You are a helpful assistant.
Use the context below to answer the question.

Context:
{context}

Question:
{query}

Answer:
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

# ----------------------------
# 7. Test
# ----------------------------
query = "What is RAG and how does it work?"
answer = generate_answer(query)

print("\nQ:", query)
print("\nA:", answer)

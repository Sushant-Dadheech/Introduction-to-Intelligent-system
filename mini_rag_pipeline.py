"""
Mini RAG (Retrieval-Augmented Generation) Pipeline from scratch
Demonstrates how RAG works mathematically under the hood using TF-IDF and
a local Ollama API - without relying on heavy abstractions like LangChain.

Requirements:
- scikit-learn
- requests
- A local Ollama server running on port 11434 with 'qwen2.5-coder:7b' or similar model.
"""

import time
import sys
import json
import requests
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Animations ---
def typing_print(text, delay=0.015):
    """Outputs text with a typewriter effect."""
    for char in str(text):
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

def loading_animation(message="Loading", dots=3, speed=0.3):
    """Displays a simple loading animation."""
    sys.stdout.write(message)
    sys.stdout.flush()
    for _ in range(dots):
        time.sleep(speed)
        sys.stdout.write(".")
        sys.stdout.flush()
    print()

# --- 1. The Knowledge Base (The "Vector DB") ---
# Imagine this is a database containing chunked text files, PDFs, or web pages.
knowledge_base = [
    "Antigravity is an advanced AI agent architecture capable of autonomous coding and automation workflows.",
    "Karnavati University is an educational institution located in Gandhinagar, Gujarat, India.",
    "TF-IDF stands for Term Frequency-Inverse Document Frequency. It is a mathematical statistic intended to reflect how important a word is to a document in a collection.",
    "RAG (Retrieval-Augmented Generation) retrieves relevant facts from a local knowledge base and injects them into an LLM's prompt to reduce hallucinations.",
    "Ollama is an open-source tool that lets you run large language models locally on your own GPU without sending data to the cloud.",
    "Sushant Dadheech is an AI & Machine Learning student passionate about LLMs, Automations, and building Vibe Coder systems."
]

# --- 2. The Retriever (Mathematical Search) ---
typing_print("=== 🧩 Mini RAG Pipeline Initialization ===", delay=0.03)
loading_animation("📥 Indexing knowledge base", dots=4, speed=0.2)

# Convert all text documents into mathematical vectors
vectorizer = TfidfVectorizer()
document_vectors = vectorizer.fit_transform(knowledge_base)

def retrieve_context(user_query, top_k=1):
    """Finds the most mathematically similar documents to the user's query."""
    # Convert the query into a vector in the same mathematical space
    query_vector = vectorizer.transform([user_query])
    
    # Calculate Cosine Similarity between the query and all documents
    similarities = cosine_similarity(query_vector, document_vectors)[0]
    
    # Get the indices of the highest scoring documents
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    # Only return it if it has some similarity (avoid totally random answers if nothing matches)
    if similarities[top_indices[0]] == 0:
        return "No relevant context found in the database."
        
    retrieved_texts = [knowledge_base[idx] for idx in top_indices]
    return " ".join(retrieved_texts)

# --- 3. The Generator (Local LLM Hook) ---
def generate_answer(query, context, model="qwen2.5-coder:7b"):
    """Injects the context into the prompt and sends it to local Ollama API."""
    prompt = f"""You are a helpful AI assistant. Answer the user's question ONLY using the context provided below. If the context does not contain the answer, say "I don't know based on the provided context."

Context: {context}

Question: {query}
Answer:"""

    url = "http://localhost:11434/api/generate"
    data = {"model": model, "prompt": prompt, "stream": True}
    
    try:
        response = requests.post(url, json=data, stream=True)
        response.raise_for_status()
        
        # Stream the response chunk by chunk as Ollama generates it
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                sys.stdout.write(chunk['response'])
                sys.stdout.flush()
                if chunk.get('done'):
                    print("\n")
                    break
    except requests.exceptions.ConnectionError:
         typing_print("\n❌ Error: Could not connect to Ollama. Make sure 'ollama serve' is running on port 11434.", delay=0.02)
    except Exception as e:
         typing_print(f"\n❌ Error during generation: {str(e)}", delay=0.02)


# --- 4. The Main Application ---
typing_print("✅ System Ready! Ask a question based on the custom knowledge base.", delay=0.02)
typing_print("(Hint: Ask about Antigravity, Karnavati University, RAG, Ollama, or TF-IDF)\n", delay=0.01)

while True:
    try:
        user_question = input("🧑 You: ")
        if user_question.lower() in ['exit', 'quit']:
            break
            
        print()
        loading_animation("🔍 Retrieving context (TF-IDF Cosine Similarity)", dots=3, speed=0.1)
        
        # Retrieve Phase
        relevant_context = retrieve_context(user_question)
        typing_print(f"📄 Retrieved Fact: {relevant_context}", delay=0.01)
        
        # Generator Phase
        print("\n🤖 Assistant: ", end="")
        generate_answer(user_question, relevant_context)
        print("-" * 50)
        
    except KeyboardInterrupt:
        break
        
typing_print("\n👋 RAG Pipeline shutting down.", delay=0.03)

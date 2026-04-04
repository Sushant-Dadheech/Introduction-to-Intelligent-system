"""
Transformers: Self-Attention from Scratch
Demonstrates the mathematics behind Large Language Models using pure Numpy matrix multiplication.
No PyTorch/TensorFlow hiding the logic!
"""

import time
import sys
import numpy as np

# ==========================================
# 🎨 Console Visuals
# ==========================================
def typing_print(text, delay=0.015):
    for char in str(text):
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

def loading_animation(message="Calculating", dots=3, speed=0.3):
    sys.stdout.write(message)
    sys.stdout.flush()
    for _ in range(dots):
        time.sleep(speed)
        sys.stdout.write(".")
        sys.stdout.flush()
    print()

# ==========================================
# 🧠 Mathematics Helpers
# ==========================================
def softmax(x):
    """Compute softmax values for each row."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True)) 
    return e_x / e_x.sum(axis=-1, keepdims=True)

# ==========================================
# 🚀 Self-Attention Implementation
# ==========================================
def run_attention():
    typing_print("=== 🤖 Transformers: Self-Attention from Scratch ===", delay=0.03)
    
    # 1. Provide an Input Sequence
    words = ["AI", "is", "awesome"]
    seq_length = len(words)
    embed_dim = 4  # D_model
    head_dim = 3   # D_k (Dimension of Keys/Queries/Values)
    
    typing_print(f"\n1️⃣  Input Sentence: '{' '.join(words)}'")
    
    # Generate random mock Word Embeddings [Sequence Length, Embedding Dimension]
    np.random.seed(42)
    X = np.round(np.random.randn(seq_length, embed_dim), 2)
    
    print("\n[Input Embeddings Matrix 'X' (3 words, 4 dimensions)]:")
    print(X)
    
    time.sleep(1)
    loading_animation("\n2️⃣  Initializing Weights for Query, Key, Value")
    
    # Initialize Random Weight Matrices [Embedding Dimension, Head Dimension]
    W_Q = np.round(np.random.randn(embed_dim, head_dim), 2)
    W_K = np.round(np.random.randn(embed_dim, head_dim), 2)
    W_V = np.round(np.random.randn(embed_dim, head_dim), 2)
    
    # 3. Calculate Q, K, and V matrices by projecting inputs (X * W)
    Q = np.dot(X, W_Q)
    K = np.dot(X, W_K)
    V = np.dot(X, W_V)
    
    typing_print("\n3️⃣  Computed Query (Q), Key (K), and Value (V) Matrices!")
    print("Q = X • W_Q:")
    print(np.round(Q, 2))
    
    time.sleep(1)
    loading_animation("\n4️⃣  Calculating Attention Scores (Q • K^T)")
    
    # 4. Calculate raw attention scores
    raw_scores = np.dot(Q, K.T)
    
    print("\n[Raw Attention Scores]:")
    print(np.round(raw_scores, 2))
    
    time.sleep(1)
    loading_animation("\n5️⃣  Scaling down by sqrt(d_k) and applying Softmax")
    
    # 5. Apply Scale (Divide by square root of dimension to stabilize gradients)
    scaled_scores = raw_scores / np.sqrt(head_dim)
    
    # 6. Apply Softmax to get Attention Probabilities (Weights)
    attention_weights = softmax(scaled_scores)
    
    print("\n[Attention Weights Matrix (Percentages)]:")
    print(np.round(attention_weights, 3))
    
    time.sleep(1)
    loading_animation("\n6️⃣  Multiplying Attention Weights • Value Matrix (V)")
    
    # 7. Final Context-Aware Embeddings!
    context_output = np.dot(attention_weights, V)
    
    typing_print("\n✅ Final Context-Aware Outputs for each word:")
    print(np.round(context_output, 2))
    
    typing_print("\n🎉 Self-Attention calculation complete! Each word now mathematically 'pays attention' to its neighbors.", delay=0.02)

if __name__ == "__main__":
    run_attention()

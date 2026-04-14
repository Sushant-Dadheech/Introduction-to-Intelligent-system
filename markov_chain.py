"""
Markov Chain Text Generator
A probabilistic Natural Language Processing (NLP) algorithm that generates 
new sentences by learning the transition probabilities between words in a corpus.
"""

import random
import time
import sys

def typing_print(text, delay=0.03):
    """Outputs text with a typewriter effect."""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

# A fun, philosophical corpus for our Markov Chain to learn from
CORPUS = [
    "artificial intelligence is the future of computing",
    "the future of humanity depends on intelligent systems",
    "intelligent systems learn from data to make decisions",
    "data is the new oil of the digital economy",
    "computing power doubles every two years",
    "the digital economy feeds on big data",
    "artificial intelligence learns from big data",
    "decisions made by artificial intelligence can change the world",
    "the world of computing is vast and complex"
]

def build_markov_model(corpus):
    """Builds a transition matrix (dictionary) from the text corpus."""
    model = {}
    
    for sentence in corpus:
        words = sentence.lower().split()
        for i in range(len(words) - 1):
            current_word = words[i]
            next_word = words[i + 1]
            
            if current_word not in model:
                model[current_word] = {}
                
            if next_word not in model[current_word]:
                model[current_word][next_word] = 0
                
            model[current_word][next_word] += 1
            
    # Convert counts to probabilities
    for current_word, transitions in model.items():
        total_transitions = sum(transitions.values())
        for next_word in transitions:
            model[current_word][next_word] /= total_transitions
            
    return model

def generate_text(model, start_word, length=10):
    """Generates new text using the trained Markov Chain model."""
    current_word = start_word.lower()
    
    if current_word not in model:
        return "Error: Start word not in vocabulary."
        
    sentence = [current_word]
    
    for _ in range(length - 1):
        if current_word not in model:
            break # Reached a word with no outward transitions
            
        transitions = model[current_word]
        next_words = list(transitions.keys())
        probabilities = list(transitions.values())
        
        # Choose the next word based on probability distribution
        current_word = random.choices(next_words, weights=probabilities)[0]
        sentence.append(current_word)
        
    return " ".join(sentence).capitalize() + "."

def main():
    typing_print("=== 🧠 Generative AI: Markov Chain Text Generator ===", delay=0.04)
    typing_print("Training the NLP model on our knowledge corpus...", delay=0.03)
    
    time.sleep(0.5)
    model = build_markov_model(CORPUS)
    typing_print(f"Model trained! Vocabulary size: {len(model)} words.\n", delay=0.03)
    
    # Let's generate a few sentences!
    start_words = ["artificial", "the", "data", "intelligent"]
    
    for word in start_words:
        typing_print(f"Generating sentence starting with '{word}':", delay=0.02)
        generated_sentence = generate_text(model, start_word=word, length=8)
        
        # Output the generated sentence slowly to simulate an LLM responding
        sys.stdout.write("  🤖 -> ")
        for char in generated_sentence:
            sys.stdout.write(char)
            sys.stdout.flush()
            time.sleep(0.05)
        print("\n")
        time.sleep(0.5)

if __name__ == "__main__":
    main()

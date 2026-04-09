"""
Medical Expert System (Backward Chaining) - Part 2
An expert system that diagnoses diseases by forming a hypothesis 
and asking the user questions to verify it using backward chaining.
"""

import time
import sys

def typing_print(text, delay=0.03):
    """Outputs text with a typewriter effect."""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

knowledge_base = {
    "covid": ["fever", "cough", "loss of smell", "tiredness"],
    "flu": ["fever", "headache", "body pain", "cough"],
    "malaria": ["fever", "chills", "sweating"],
    "dengue": ["fever", "joint pain", "rash", "vomiting"]
}

def backward_chaining(disease, rules, asked_questions, known_symptoms):
    """Verifies a disease by asking about its symptoms recursively."""
    if disease not in rules:
        return False
        
    symptoms = rules[disease]
    for symptom in symptoms:
        if symptom in known_symptoms:
            continue # already confirmed this symptom
        elif symptom in asked_questions and symptom not in known_symptoms:
            return False # already denied this symptom
        
        # Ask the user if we don't know
        typing_print(f"👉 Do you have this symptom: '{symptom}'? (y/n): ", delay=0.02)
        response = input().strip().lower()
        asked_questions.add(symptom)
        
        if response == 'y':
            known_symptoms.add(symptom)
        else:
            return False # if any symptom is missing, hypothesis fails
            
    return True

def main():
    typing_print("=== 🏥 Medical Expert System (Backward Chaining) ===", delay=0.04)
    typing_print("The system will formulate a hypothesis and ask questions to verify it.\n", delay=0.03)
    
    asked_questions = set()
    known_symptoms = set()
    diagnosed = False
    
    # The system forms hypotheses (diseases) and tests them
    for hypothesis in knowledge_base.keys():
        typing_print(f"\n[Hypothesis] Testing for {hypothesis.capitalize()}...", delay=0.04)
        time.sleep(0.5)
        
        if backward_chaining(hypothesis, knowledge_base, asked_questions, known_symptoms):
            typing_print("\n--- 📋 Diagnosis Result ---", delay=0.05)
            typing_print(f" Based on your answers, you likely have: ⚠️ {hypothesis.capitalize()}", delay=0.04)
            typing_print(" Please consult a real doctor for professional advice.", delay=0.04)
            diagnosed = True
            break
        else:
            typing_print(f"❌ Hypothesis for {hypothesis.capitalize()} failed.", delay=0.02)
            
    if not diagnosed:
        typing_print("\n--- 📋 Diagnosis Result ---", delay=0.05)
        typing_print("Could not diagnose the disease with the given knowledge base.", delay=0.04)

if __name__ == "__main__":
    main()

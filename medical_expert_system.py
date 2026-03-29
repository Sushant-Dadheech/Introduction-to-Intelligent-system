"""
Medical Expert System (Forward Chaining) - Part 1
A simple expert system that diagnoses diseases based on user symptoms.
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

def loading_animation(message="Analyzing symptoms"):
    """Displays a loading animation with periods."""
    sys.stdout.write(message)
    sys.stdout.flush()
    for _ in range(5):
        time.sleep(0.4)
        sys.stdout.write(".")
        sys.stdout.flush()
    print("\n")

rules = {
    "cold":    {"cough", "sneezing", "runny nose"},
    "flu":     {"fever", "headache", "body pain", "cough"},
    "malaria": {"fever", "chills", "sweating"},
    "typhoid": {"fever", "stomach pain", "weakness"},
    "covid":   {"fever", "cough", "loss of smell", "tiredness"}
}

def forward_chaining(facts, rules):
    conclusions = []
    for disease, symptoms in rules.items():
        if symptoms.issubset(facts):
            conclusions.append(disease)
    return conclusions

typing_print("=== 🏥 Simple Medical Expert System ===", delay=0.05)
print("\nAvailable symptoms:")
print("cough, sneezing, runny nose, fever, headache, body pain,")
print("chills, sweating, stomach pain, weakness, loss of smell, tiredness\n")

user_input = input("👉 Enter your symptoms separated by commas: ")
user_symptoms = set(sym.strip().lower() for sym in user_input.split(","))

print()
loading_animation("🔬 Analyzing your symptoms in the knowledge base")

possible_diseases = forward_chaining(user_symptoms, rules)

typing_print("--- 📋 Diagnosis Result ---", delay=0.05)
if possible_diseases:
    typing_print("Based on your symptoms, you may have:", delay=0.04)
    for d in possible_diseases:
        time.sleep(0.3)
        typing_print(f" ⚠️  {d.capitalize()}", delay=0.06)
else:
    typing_print("No exact matching disease found in the database. 🩺 Please consult a real doctor.", delay=0.04)

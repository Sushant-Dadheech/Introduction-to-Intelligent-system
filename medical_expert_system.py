"""
Medical Expert System (Forward Chaining) - Part 1
A simple expert system that diagnoses diseases based on user symptoms.
"""

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

print("=== Simple Medical Expert System ===")
print("Available symptoms:")
print("cough, sneezing, runny nose, fever, headache, body pain,")
print("chills, sweating, stomach pain, weakness, loss of smell, tiredness\n")

user_input = input("Enter your symptoms separated by commas: ")
user_symptoms = set(sym.strip() for sym in user_input.split(","))

possible_diseases = forward_chaining(user_symptoms, rules)

print("\n--- Diagnosis Result ---")
if possible_diseases:
    print("Based on your symptoms, you may have:")
    for d in possible_diseases:
        print(f" - {d.capitalize()}")
else:
    print("No matching disease found. Please consult a doctor.")

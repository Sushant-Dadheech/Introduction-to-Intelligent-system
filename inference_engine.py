# Question One - Inference Engine Implementation
# Sushant Dadheech
# ku_id - Ku2407u814

# ============================================
# Example 1: Medical Diagnosis Inference
# ============================================

class InferenceEngine:
    def __init__(self):
        self.rules = []
        self.facts = set()

    def add_rule(self, conditions, result):
        self.rules.append({'if': conditions, 'then': result})

    def add_fact(self, fact):
        self.facts.add(fact)

    def run(self):
        new_fact_discovered = True
        while new_fact_discovered:
            new_fact_discovered = False
            for rule in self.rules:
                if all(cond in self.facts for cond in rule['if']):
                    if rule['then'] not in self.facts:
                        print(f"Logic triggered: Because {rule['if']} is true, then {rule['then']} is true.")
                        self.facts.add(rule['then'])
                        new_fact_discovered = True
        return self.facts


# --- Example 1: Medical Diagnosis ---
print("=" * 50)
print("Example 1: Medical Diagnosis Inference")
print("=" * 50)

engine = InferenceEngine()

# Rules
engine.add_rule(["Sneeze", "Fever"], "Flu")
engine.add_rule(["Flu"], "Stay Home")
engine.add_rule(["Rain", "Wind"], "Stay Home")

# Facts
engine.add_fact("Sneeze")
engine.add_fact("Fever")

final_knowledge_base = engine.run()
print(f"\nFinal Facts Known: {final_knowledge_base}")


# ============================================
# Example 2: Daily Decision Inference
# ============================================

print("\n" + "=" * 50)
print("Example 2: Daily Decision Inference")
print("=" * 50)

engine2 = InferenceEngine()

# Rules
engine2.add_rule(["Hungry", "Has Money"], "Buy Food")
engine2.add_rule(["Buy Food"], "Eat")
engine2.add_rule(["Eat"], "Happy")

# Facts
engine2.add_fact("Hungry")
engine2.add_fact("Has Money")

final_knowledge_base2 = engine2.run()
print(f"\nFinal Facts Known: {final_knowledge_base2}")

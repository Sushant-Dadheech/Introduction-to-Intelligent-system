import os
import re

directory = r"c:\Users\sushant\Desktop\Github Daily Posting\repo"

animation_code = """
import time
import sys

def typing_print(text, delay=0.01):
    for char in str(text):
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

def loading_animation(message="Loading"):
    sys.stdout.write(message)
    sys.stdout.flush()
    for _ in range(3):
        time.sleep(0.3)
        sys.stdout.write(".")
        sys.stdout.flush()
    print("\\n")
"""

files = [
    "bayesian_network.py",
    "comparative_analysis.py",
    "heuristic_search.py",
    "inference_engine.py",
    "ml_algorithms.py",
    "neural_network_training_part1.py",
    "neural_network_training_part2.py",
    "q_learning.py",
    "tsp_algorithms.py"
]

for filename in files:
    filepath = os.path.join(directory, filename)
    if not os.path.exists(filepath):
        continue
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Skip if already modified
    if "def typing_print" in content:
        continue
    
    # Insert animation code after the docstring or at the top
    if content.startswith('"""'):
        parts = content.split('"""', 2)
        if len(parts) >= 3:
            new_content = '"""' + parts[1] + '"""\n' + animation_code + parts[2]
        else:
            new_content = animation_code + "\n" + content
    else:
        new_content = animation_code + "\n" + content
        
    # Pattern 1: print("Literal String") -> typing_print("Literal String")
    new_content = re.sub(r'print\(\s*(["\'][^"\'\{\}]+["\'])\s*\)', r'typing_print(\1)', new_content)

    # Pattern 2: print("=" * N) -> typing_print("=" * N, delay=0.002)
    new_content = re.sub(r'print\(\s*(["\'][=*-]+["\']\s*\*\s*\d+)\s*\)', r'typing_print(\1, delay=0.002)', new_content)
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(new_content)
        
print("Successfully animated files.")

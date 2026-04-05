"""
Interactive CLI Hub for Introduction to Intelligent Systems
A centralized dashboard to easily execute all 14 AI/ML practicals!
"""

import time
import sys
import subprocess
import os

# ==========================================
# 🎨 Console Visuals
# ==========================================
def typing_print(text, delay=0.01):
    for char in str(text):
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

# ==========================================
# 📚 Project Database
# ==========================================
PROJECTS = {
    "1": {"name": "Travelling Salesman Problem (BFS, DFS, A*)", "file": "tsp_algorithms.py"},
    "2": {"name": "Heuristic Search (Simulated Annealing, GA)", "file": "heuristic_search.py"},
    "3": {"name": "ML Models (Supervised & Unsupervised)", "file": "ml_algorithms.py"},
    "4": {"name": "Inference Engine (Forward Chaining)", "file": "inference_engine.py"},
    "5": {"name": "Neural Network Training - Part 1", "file": "neural_network_training_part1.py"},
    "6": {"name": "Neural Network Training - Part 2", "file": "neural_network_training_part2.py"},
    "7": {"name": "Q-Learning Algorithm (RL)", "file": "q_learning.py"},
    "8": {"name": "Medical Expert System", "file": "medical_expert_system.py"},
    "9": {"name": "Bayesian Network (pgmpy)", "file": "bayesian_network.py"},
    "10": {"name": "Comparative Analysis (K-Means, LR, RF)", "file": "comparative_analysis.py"},
    "11": {"name": "Mini RAG Pipeline (Ollama)", "file": "mini_rag_pipeline.py"},
    "12": {"name": "Constraint Satisfaction Problem (Graph Coloring)", "file": "csp_graph_coloring.py"},
    "13": {"name": "Mini AI Agent (Reasoning + Acting)", "file": "mini_ai_agent.py"},
    "14": {"name": "Transformers: Self-Attention from Scratch", "file": "self_attention_scratch.py"},
}

def display_menu():
    clear_screen()
    print("=" * 65)
    typing_print(" 🤖 INTRODUCTION TO INTELLIGENT SYSTEMS - LAB DASHBOARD ", delay=0.005)
    print("=" * 65)
    
    for key, value in PROJECTS.items():
        print(f" [{key.rjust(2)}] {value['name']}")
        
    print("-" * 65)
    print(" [q]  Quit Dashboard")
    print("=" * 65)

def main_loop():
    while True:
        display_menu()
        choice = input("\nSelect a practical to execute (1-14, or q): ").strip().lower()
        
        if choice in ['q', 'quit', 'exit']:
            typing_print("\n👋 Shutting down Dashboard. Goodbye!", delay=0.02)
            break
            
        if choice in PROJECTS:
            script_to_run = PROJECTS[choice]["file"]
            typing_print(f"\n🚀 Launching: {PROJECTS[choice]['name']}...", delay=0.015)
            time.sleep(0.5)
            clear_screen()
            
            try:
                # Execute the child python script
                # Using sys.executable ensures the script runs in the exact same colab/local env
                subprocess.run([sys.executable, script_to_run])
            except Exception as e:
                print(f"\n❌ Failed to execute {script_to_run}. Error: {e}")
                
            print("\n" + "=" * 50)
            input("Press Enter to return to the Main Menu...")
        else:
            print("⚠️ Invalid selection. Please enter a valid number or 'q'.")
            time.sleep(1)

if __name__ == "__main__":
    main_loop()

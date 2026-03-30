
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
    typing_print("\n")

# ============================================================
#: Heuristic Search Algorithms
# Name: Sushant Dadheech
# Algorithms: Simulated Annealing & Genetic Algorithm
# ============================================================

import math
import random
import matplotlib.pyplot as plt

# ---- Objective Function (Common for both algorithms) ----
def objective(x):
    return x**2 + 10 * math.sin(x)

# ============================================================
# 1️⃣ SIMULATED ANNEALING
# ============================================================
def simulated_annealing():
    current = random.uniform(-10, 10)
    best = current
    T = 1000        # Initial temperature
    Tmin = 1e-3     # Minimum temperature
    alpha = 0.9     # Cooling rate
    history = []

    while T > Tmin:
        new = current + random.uniform(-1, 1)
        delta = objective(new) - objective(current)

        if delta < 0 or random.random() < math.exp(-delta / T):
            current = new

        if objective(current) < objective(best):
            best = current

        history.append(objective(best))
        T *= alpha

    return best, objective(best), history

best_x, best_val, history = simulated_annealing()

typing_print("=" * 45, delay=0.002)
typing_print("🔵 Simulated Annealing Result")
print(f"   Best x      : {best_x}")
print(f"   Minimum value: {best_val}")
typing_print("=" * 45, delay=0.002)

plt.plot(history)
plt.xlabel("Iterations")
plt.ylabel("Best Objective Value")
plt.title("Simulated Annealing Convergence")
plt.grid(True)
plt.show()

# ============================================================
# 2️⃣ GENETIC ALGORITHM
# ============================================================
def fitness(x):
    return -objective(x)

def genetic_algorithm():
    population_size = 30
    generations = 100
    mutation_rate = 0.1
    population = [random.uniform(-10, 10) for _ in range(population_size)]
    history = []

    for _ in range(generations):
        population = sorted(population, key=objective)
        history.append(objective(population[0]))

        # Selection (Top 50%)
        selected = population[:population_size // 2]
        children = []

        while len(children) < population_size:
            parent1, parent2 = random.sample(selected, 2)
            child = (parent1 + parent2) / 2  # Crossover

            # Mutation
            if random.random() < mutation_rate:
                child += random.uniform(-1, 1)

            children.append(child)

        population = children

    best = min(population, key=objective)
    return best, objective(best), history

best_x, best_val, history = genetic_algorithm()

typing_print("🟢 Genetic Algorithm Result")
print(f"   Best x      : {best_x}")
print(f"   Minimum value: {best_val}")
typing_print("=" * 45, delay=0.002)

plt.plot(history)
plt.xlabel("Generations")
plt.ylabel("Best Objective Value")
plt.title("Genetic Algorithm Convergence")
plt.grid(True)
plt.show()

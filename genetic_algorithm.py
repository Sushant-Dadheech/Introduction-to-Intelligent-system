"""
Genetic Algorithm - String Evolution
A simple AI algorithm that evolves a population of random strings 
to match a target phrase using selection, crossover, and mutation.
"""

import random
import string
import time
import sys

# Target string to evolve into
TARGET = "Artificial Intelligence"
POPULATION_SIZE = 200
MUTATION_RATE = 0.05
GENES = string.ascii_letters + " "

def typing_print(text, delay=0.03, end="\n"):
    """Outputs text with a typewriter effect."""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    sys.stdout.write(end)
    sys.stdout.flush()

def create_individual():
    """Create a random string of the same length as the target."""
    return "".join(random.choice(GENES) for _ in range(len(TARGET)))

def calculate_fitness(individual):
    """Calculate how many characters match the target."""
    fitness = sum(1 for expected, actual in zip(TARGET, individual) if expected == actual)
    return fitness

def crossover(parent1, parent2):
    """Combine two parents to create a child."""
    split = random.randint(0, len(TARGET) - 1)
    child = parent1[:split] + parent2[split:]
    return child

def mutate(individual):
    """Randomly mutate some characters based on the mutation rate."""
    individual = list(individual)
    for i in range(len(individual)):
        if random.random() < MUTATION_RATE:
            individual[i] = random.choice(GENES)
    return "".join(individual)

def main():
    typing_print("=== 🧬 Genetic Algorithm: String Evolution ===", delay=0.05)
    typing_print(f"Target Phrase : '{TARGET}'", delay=0.04)
    typing_print(f"Population    : {POPULATION_SIZE}", delay=0.04)
    typing_print(f"Mutation Rate : {MUTATION_RATE * 100}%\n", delay=0.04)
    time.sleep(1)

    # Initialize population
    population = [create_individual() for _ in range(POPULATION_SIZE)]
    generation = 1
    found = False

    while not found:
        # Sort population by fitness
        population = sorted(population, key=calculate_fitness, reverse=True)
        
        best_individual = population[0]
        best_fitness = calculate_fitness(best_individual)
        
        # Display progress
        sys.stdout.write(f"\rGeneration {generation:4d} | Best Match: '{best_individual}' | Fitness: {best_fitness}/{len(TARGET)}")
        sys.stdout.flush()
        time.sleep(0.05)
        
        if best_fitness == len(TARGET):
            found = True
            break
            
        # Select the top 10% of the population as parents
        parents = population[:int(POPULATION_SIZE * 0.1)]
        
        # Create next generation
        next_generation = []
        # Keep the best individual (Elitism)
        next_generation.append(best_individual)
        
        while len(next_generation) < POPULATION_SIZE:
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)
            child = crossover(parent1, parent2)
            child = mutate(child)
            next_generation.append(child)
            
        population = next_generation
        generation += 1

    print("\n")
    typing_print("🎉 Evolution Complete!", delay=0.05)
    typing_print(f"Target '{TARGET}' reached in {generation} generations.", delay=0.05)

if __name__ == "__main__":
    main()

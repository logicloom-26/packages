#Knapsack Problem using a Genetic Algorithm: Viva Notes
#What is the problem?
'''You have 8 items, each with a weight and a value, and a
bag that can carry at most 20 kg. You have to choose which
items to take so that the total value is as large as possible
without going over the weight limit. This is the 0/1 knapsack problem,
because each item is either taken or left (no partial items).'''

#What does the code do?
'''It uses a Genetic Algorithm (GA) to search for the best selection of items.
It mimics natural evolution: good solutions survive and combine, and new variations are
introduced by random changes.'''

import random

# ---------------- Problem definition ----------------
# Each item = (weight, value)
items = [
    (2, 10),
    (3, 5),
    (4, 15),
    (5, 7),
    (9, 20),
    (7, 8),
    (1, 3),
    (6, 12),
]
CAPACITY = 20            # knapsack weight limit

POPULATION_SIZE = 10
MUTATION_RATE = 0.10
GENERATIONS = 200
CHROMOSOME_LENGTH = len(items)      # one bit per item: 1 = take it, 0 = skip


# ---------------- GA building blocks ----------------

def random_chromosome():
    return [random.randint(0, 1) for _ in range(CHROMOSOME_LENGTH)]


def fitness(chromosome):
    total_weight = 0
    total_value = 0
    for gene, (weight, value) in zip(chromosome, items):
        if gene == 1:
            total_weight += weight
            total_value += value
    if total_weight > CAPACITY:          # invalid solution -> no fitness
        return 0
    return total_value


def tournament_selection(population, k=3):
    contenders = random.sample(population, k)
    return max(contenders, key=fitness).copy()


def crossover(parent1, parent2):
    point = random.randint(1, CHROMOSOME_LENGTH - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2


def mutation(chromosome):
    for i in range(len(chromosome)):
        if random.random() < MUTATION_RATE:
            chromosome[i] = 1 - chromosome[i]      # flip bit
    return chromosome


def describe(chromosome):
    chosen = [items[i] for i in range(CHROMOSOME_LENGTH) if chromosome[i] == 1]
    total_weight = sum(w for w, v in chosen)
    total_value = sum(v for w, v in chosen)
    return chosen, total_weight, total_value


# ---------------- Genetic Algorithm ----------------

population = [random_chromosome() for _ in range(POPULATION_SIZE)]

best_overall = max(population, key=fitness)
best_found_at_generation = 0

for generation in range(GENERATIONS):
    new_population = []

    # Elitism: keep the best chromosome seen so far
    best = max(population, key=fitness)
    if fitness(best) > fitness(best_overall):
        best_overall = best.copy()
        best_found_at_generation = generation + 1
    new_population.append(best.copy())

    while len(new_population) < POPULATION_SIZE:
        parent1 = tournament_selection(population)
        parent2 = tournament_selection(population)
        child1, child2 = crossover(parent1, parent2)
        child1 = mutation(child1)
        child2 = mutation(child2)
        new_population.append(child1)
        if len(new_population) < POPULATION_SIZE:
            new_population.append(child2)

    population = new_population
    # (no per-generation printing -- only the final result is shown below)

print("\n" + "=" * 40)
print("BEST SOLUTION FOUND")
print("=" * 40)
chosen, total_weight, total_value = describe(best_overall)
print("Chromosome:", best_overall)
print("Found at generation:", best_found_at_generation, "out of", GENERATIONS)
print("\nItems chosen (weight, value):")
for item in chosen:
    print("  ", item)
print("\nTotal weight:", total_weight, "/ Capacity:", CAPACITY)
print("Total value :", total_value)

import random

target = "FYBSC"
POPULATION_SIZE = 100      # bigger population = more letter variety per generation
MUTATION_RATE = 0.20       # higher mutation finds correct letters faster
GENERATIONS = 1000
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

# Initial population (random, no user input)
population = [[random.choice(LETTERS) for _ in range(len(target))]
              for _ in range(POPULATION_SIZE)]

# Fitness function
def fitness(chromosome):
    return sum(1 for i in range(len(target)) if chromosome[i] == target[i])

# Roulette wheel selection
def roulette_selection(population):
    fitness_values = [fitness(ch) for ch in population]
    total = sum(fitness_values)
    if total == 0:
        return random.choice(population).copy()
    r = random.uniform(0, total)
    cumulative = 0
    for chromosome, fit in zip(population, fitness_values):
        cumulative += fit
        if cumulative >= r:
            return chromosome.copy()
    return population[-1].copy()

# Crossover
def crossover(parent1, parent2):
    point = random.randint(1, len(target) - 1)
    return (parent1[:point] + parent2[point:],
            parent2[:point] + parent1[point:])

# Mutation
def mutation(chromosome):
    for i in range(len(chromosome)):
        if random.random() < MUTATION_RATE:
            chromosome[i] = random.choice(LETTERS)
    return chromosome

# Genetic Algorithm
for generation in range(GENERATIONS):
    new_population = []

    # Elitism
    best = max(population, key=fitness)
    new_population.append(best.copy())

    while len(new_population) < POPULATION_SIZE:
        parent1 = roulette_selection(population)
        parent2 = roulette_selection(population)
        child1, child2 = crossover(parent1, parent2)
        new_population.append(mutation(child1))
        if len(new_population) < POPULATION_SIZE:
            new_population.append(mutation(child2))

    population = new_population
    best = max(population, key=fitness)

    # Print only the best of each generation
    print(f"Generation {generation + 1}: {''.join(best)}  Fitness: {fitness(best)}")

    if fitness(best) == len(target):
        print("\nTarget Found:", "".join(best))
        break

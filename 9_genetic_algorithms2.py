
#Tournament vs Rank Selection (INDIA)
import random

target = "INDIA"
POPULATION_SIZE = 100
MUTATION_RATE = 0.20
GENERATIONS = 1000
TOURNAMENT_SIZE = 3
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def fresh_population():
    # Random starting population (new one for every run)
    return [[random.choice(LETTERS) for _ in range(len(target))]
            for _ in range(POPULATION_SIZE)]


# Fitness Function
def fitness(chromosome):
    return sum(1 for i in range(len(target)) if chromosome[i] == target[i])


# ---------------- SELECTION METHODS ----------------
def tournament_selection(population):
    contenders = random.sample(population, TOURNAMENT_SIZE)
    return max(contenders, key=fitness).copy()


def rank_selection(population):
    ranked = sorted(population, key=fitness)
    ranks = list(range(1, len(ranked) + 1))
    total_rank = sum(ranks)
    r = random.uniform(0, total_rank)
    cumulative = 0
    for chromosome, rank in zip(ranked, ranks):
        cumulative += rank
        if cumulative >= r:
            return chromosome.copy()
    return ranked[-1].copy()


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


# Runs the GA fully (no per-generation printing) and returns
# how many generations it took, and the best string found.
def run_ga(select_fn):
    population = fresh_population()
    for generation in range(GENERATIONS):
        new_population = []

        # Elitism
        best = max(population, key=fitness)
        new_population.append(best.copy())

        while len(new_population) < POPULATION_SIZE:
            parent1 = select_fn(population)
            parent2 = select_fn(population)
            child1, child2 = crossover(parent1, parent2)
            new_population.append(mutation(child1))
            if len(new_population) < POPULATION_SIZE:
                new_population.append(mutation(child2))

        population = new_population
        best = max(population, key=fitness)

        if fitness(best) == len(target):
            return generation + 1, "".join(best)

    # did not converge within GENERATIONS
    return GENERATIONS, "".join(max(population, key=fitness))


# ---------------- Run and print FINAL result only ----------------
methods = {
    "Tournament Selection": tournament_selection,
    "Rank Selection": rank_selection,
}

print("=" * 50)
print("FINAL RESULT: Generations to find target '{}'".format(target))
print("=" * 50)
print("{:<28}{:<15}{}".format("Selection Method", "Generations", "Result"))
print("-" * 50)
for name, select_fn in methods.items():
    generations_taken, result = run_ga(select_fn)
    print("{:<28}{:<15}{}".format(name, generations_taken, result))

#--------------------------------------------------------------------------------------------------
#Compare all selections (INDIA)
import random
import statistics

target = "INDIA"
POPULATION_SIZE = 100
MUTATION_RATE = 0.20
GENERATIONS = 1000
TOURNAMENT_SIZE = 3
TRIALS = 100
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def fresh_population():
    # Random starting population (new one for every run)
    return [[random.choice(LETTERS) for _ in range(len(target))]
            for _ in range(POPULATION_SIZE)]


# Fitness Function
def fitness(chromosome):
    return sum(1 for i in range(len(target)) if chromosome[i] == target[i])


# ---------------- SELECTION METHODS ----------------

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


def tournament_selection(population):
    contenders = random.sample(population, TOURNAMENT_SIZE)
    return max(contenders, key=fitness).copy()


def rank_selection(population):
    ranked = sorted(population, key=fitness)
    ranks = list(range(1, len(ranked) + 1))
    total_rank = sum(ranks)
    r = random.uniform(0, total_rank)
    cumulative = 0
    for chromosome, rank in zip(ranked, ranks):
        cumulative += rank
        if cumulative >= r:
            return chromosome.copy()
    return ranked[-1].copy()


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


def run_ga(select_fn):
    population = fresh_population()
    for generation in range(GENERATIONS):
        new_population = []

        # Elitism
        best = max(population, key=fitness)
        new_population.append(best.copy())

        while len(new_population) < POPULATION_SIZE:
            parent1 = select_fn(population)
            parent2 = select_fn(population)
            child1, child2 = crossover(parent1, parent2)
            new_population.append(mutation(child1))
            if len(new_population) < POPULATION_SIZE:
                new_population.append(mutation(child2))

        population = new_population
        best = max(population, key=fitness)

        if fitness(best) == len(target):
            return generation + 1

    return GENERATIONS   # did not converge


# ---------------- Run comparison over many trials ----------------

methods = {
    "Roulette Wheel": roulette_selection,
    "Tournament":     tournament_selection,
    "Rank":           rank_selection,
}

results = {name: [] for name in methods}

for trial in range(TRIALS):
    for name, select_fn in methods.items():
        results[name].append(run_ga(select_fn))

print("=" * 60)
print("SELECTION METHOD COMPARISON  (target = '{}', {} trials each)".format(target, TRIALS))
print("=" * 60)
print("{:<16}{:<10}{:<8}{:<8}{}".format("Method", "Avg Gens", "Min", "Max", "StdDev"))
print("-" * 60)

summary = {}
for name, gens_list in results.items():
    avg = sum(gens_list) / len(gens_list)
    sd = statistics.stdev(gens_list)
    summary[name] = (avg, sd)
    print("{:<16}{:<10.1f}{:<8}{:<8}{:.1f}".format(name, avg, min(gens_list), max(gens_list), sd))

fastest = min(summary, key=lambda n: summary[n][0])
most_consistent = min(summary, key=lambda n: summary[n][1])

print("\n" + "=" * 60)
print("CONCLUSION")
print("=" * 60)
print("Fastest on average              :", fastest, "({:.1f} generations)".format(summary[fastest][0]))
print("Most consistent (lowest std dev):", most_consistent, "({:.1f})".format(summary[most_consistent][1]))

if fastest == most_consistent:
    print("\n'{}' is the most suitable selection method for this problem -".format(fastest))
    print("it is both the fastest on average and the most consistent across runs.")
else:
    print("\n'{}' is the most suitable selection method for this problem overall -".format(fastest))
    print("it converges fastest on average. '{}' is the more consistent".format(most_consistent))
    print("(lower variance) of the two, worth noting as a reliable alternative")
    print("if consistency matters more than raw speed.")


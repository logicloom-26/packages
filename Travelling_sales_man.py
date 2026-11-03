
#Travelling Salesman
#Summary: Find the shortest route that visits every city once and returns to the start.
#Chromosome: an ordering of the cities.
#Fitness: 1 / total route distance, so shorter routes score higher.
#Crossover and mutation: order crossover (so no city repeats) and a swap of two cities.

import random
import math

cities = {"A": (0, 0), "B": (2, 6), "C": (5, 2), "D": (6, 6),
          "E": (8, 3), "F": (1, 9), "G": (9, 9), "H": (4, 8)}
names = list(cities)
POP_SIZE = 100
MUTATION_RATE = 0.2
GENERATIONS = 200
TOURNAMENT = 5

def distance(a, b):
    return math.dist(cities[a], cities[b])

def total_distance(route):
    return sum(distance(route[i], route[(i + 1) % len(route)]) for i in range(len(route)))

def fitness(route):
    return 1 / total_distance(route)

def select(pop):
    return max(random.sample(pop, TOURNAMENT), key=fitness)

def crossover(p1, p2):
    a, b = sorted(random.sample(range(len(p1)), 2))
    child = [None] * len(p1)
    child[a:b] = p1[a:b]
    rest = [c for c in p2 if c not in child]
    for i in range(len(child)):
        if child[i] is None:
            child[i] = rest.pop(0)
    return child

def mutate(route):
    if random.random() < MUTATION_RATE:
        i, j = random.sample(range(len(route)), 2)
        route[i], route[j] = route[j], route[i]
    return route

population = [random.sample(names, len(names)) for _ in range(POP_SIZE)]
for gen in range(GENERATIONS):
    best = max(population, key=fitness)
    new_pop = [best[:]]
    while len(new_pop) < POP_SIZE:
        new_pop.append(mutate(crossover(select(population), select(population))))
    population = new_pop
    if (gen + 1) % 50 == 0:
        print(f"Generation {gen + 1}: distance = {total_distance(max(population, key=fitness)):.2f}")

best = max(population, key=fitness)
print("Best route:", " -> ".join(best + [best[0]]))
print("Total distance:", round(total_distance(best), 2))

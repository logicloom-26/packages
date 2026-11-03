#Assigning Jobs to Machines
#Summary: Assign each job to a different machine so the total cost (time) is as low as possible.
#Chromosome: a permutation, where position = job and value = machine.
#Fitness: 1 / total cost, using a cost table of job × machine.
#Crossover and mutation: order crossover and a swap, so every machine is used exactly once.
import random

# cost[job][machine] = time/cost for that machine to do that job
cost = [[9, 2, 7, 8, 6],
        [6, 4, 3, 7, 5],
        [5, 8, 1, 8, 4],
        [7, 6, 9, 4, 3],
        [8, 3, 6, 5, 2]]
N = len(cost)
POP_SIZE = 50
MUTATION_RATE = 0.2
GENERATIONS = 100
TOURNAMENT = 3

def total_cost(ch):                      # ch[job] = machine
    return sum(cost[j][ch[j]] for j in range(N))

def fitness(ch):
    return 1 / total_cost(ch)

def select(pop):
    return max(random.sample(pop, TOURNAMENT), key=fitness)

def crossover(p1, p2):
    a, b = sorted(random.sample(range(N), 2))
    child = [None] * N
    child[a:b] = p1[a:b]
    rest = [m for m in p2 if m not in child]
    for i in range(N):
        if child[i] is None:
            child[i] = rest.pop(0)
    return child

def mutate(ch):
    if random.random() < MUTATION_RATE:
        i, j = random.sample(range(N), 2)
        ch[i], ch[j] = ch[j], ch[i]
    return ch

population = [random.sample(range(N), N) for _ in range(POP_SIZE)]
for gen in range(GENERATIONS):
    best = max(population, key=fitness)
    new_pop = [best[:]]
    while len(new_pop) < POP_SIZE:
        new_pop.append(mutate(crossover(select(population), select(population))))
    population = new_pop
    if (gen + 1) % 25 == 0:
        print(f"Generation {gen + 1}: total cost = {total_cost(max(population, key=fitness))}")

best = max(population, key=fitness)
for job, machine in enumerate(best):
    print(f"Job {job + 1} -> Machine {machine + 1} (cost {cost[job][machine]})")
print("Total cost:", total_cost(best))

#Finding the Best Candidates for a Job
#Summary: From a pool of applicants, choose the group of 3 with the highest combined score.
#Chromosome: a list of 3 distinct candidate names.
#Fitness: the sum of weighted scores (skills 50%, experience 30%, interview 20%).
#Crossover and mutation: pool both parents and pick 3 distinct people. Mutation swaps one selected candidate for an outsider.


import random

# name: (skills, experience, interview)  scores out of 10
candidates = {"Amit": (8, 6, 7), "Neha": (9, 8, 9), "Rahul": (5, 4, 6),
              "Priya": (7, 9, 8), "Karan": (6, 5, 5), "Sneha": (9, 7, 8),
              "Vijay": (4, 3, 4), "Pooja": (8, 8, 7), "Rohan": (6, 7, 6),
              "Anita": (7, 6, 9)}
names = list(candidates)
WEIGHTS = (0.5, 0.3, 0.2)                # importance of skills, experience, interview
VACANCIES = 3
POP_SIZE = 50
MUTATION_RATE = 0.2
GENERATIONS = 50
TOURNAMENT = 3

def score(name):
    return sum(w * s for w, s in zip(WEIGHTS, candidates[name]))

def fitness(team):                       # team = list of selected candidate names
    return sum(score(n) for n in team)

def select(pop):
    return max(random.sample(pop, TOURNAMENT), key=fitness)

def crossover(p1, p2):
    pool = list(set(p1 + p2))
    random.shuffle(pool)
    return pool[:VACANCIES]              # always VACANCIES distinct people

def mutate(team):
    if random.random() < MUTATION_RATE:
        outsiders = [n for n in names if n not in team]
        team[random.randrange(VACANCIES)] = random.choice(outsiders)
    return team

population = [random.sample(names, VACANCIES) for _ in range(POP_SIZE)]
for gen in range(GENERATIONS):
    best = max(population, key=fitness)
    new_pop = [best[:]]
    while len(new_pop) < POP_SIZE:
        new_pop.append(mutate(crossover(select(population), select(population))))
    population = new_pop
    if (gen + 1) % 10 == 0:
        print(f"Generation {gen + 1}: best total score = {fitness(max(population, key=fitness)):.2f}")

best = max(population, key=fitness)
print("Selected candidates:")
for n in best:
    print(f"  {n}: score {score(n):.2f}")
print("Total score:", round(fitness(best), 2))

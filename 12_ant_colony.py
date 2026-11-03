# PRACTICAL 12: Ant Colony Optimization (ACO)
######################################################################
from random import choice, randrange, random

vertices = ['A', 'B', 'C', 'D']
graph = {
    'A': ['B', 'C', 'D'],
    'B': ['A', 'C', 'D'],
    'C': ['A', 'B', 'D'],
    'D': ['A', 'B', 'C'],
}
weights = {'AB': 10, 'AC': 10, 'AD': 30, 'BC': 40, 'CD': 20, 'BD': 10}
Pheromone = {}

# decay rate
p = 0.1
num_ants = 4
max_iterations = 50

def edge_key(a, b):
    """Return a consistent key for an undirected edge."""
    return ''.join(sorted(a + b))

def edge_weight(a, b):
    """Get the weight for an undirected edge."""
    key = edge_key(a, b)
    return weights.get(key, 999999)

def initializePheromone():
    """Initialize pheromone trails randomly."""
    global Pheromone
    Pheromone = {}
    for key, value in graph.items():
        for neighbor in value:
            edge = edge_key(key, neighbor)
            if edge not in Pheromone:
                Pheromone[edge] = randrange(1, 51)

def choose_next_vertex(current, visited):
    """Choose the next vertex using pheromone and heuristic value."""
    candidates = [v for v in graph[current] if v not in visited]
    if not candidates:
        return None
    scores = []
    total_score = 0.0
    for vertex in candidates:
        edge = edge_key(current, vertex)
        pheromone = Pheromone.get(edge, 1e-6)
        heuristic = 1 / edge_weight(current, vertex)
        score = pheromone * heuristic
        scores.append((vertex, score))
        total_score += score
    if total_score == 0:
        return choice(candidates)
    target = random() * total_score
    cumulative = 0.0
    for vertex, score in scores:
        cumulative += score
        if target <= cumulative:
            return vertex
    return scores[-1][0]

def tour_cost(path):
    """Compute total cost of a closed tour."""
    total = 0
    for i in range(len(path) - 1):
        total += edge_weight(path[i], path[i + 1])
    return total

def AOC(Pheromone_dict):
    """Run one ant colony optimization iteration."""
    global weights, graph
    paths = []
    best_path = None
    best_cost = float('inf')
    for _ in range(num_ants):
        start = vertices[randrange(len(vertices))]
        path = [start]
        visited = {start}
        current = start
        while len(visited) < len(vertices):
            next_vertex = choose_next_vertex(current, visited)
            if next_vertex is None:
                break
            path.append(next_vertex)
            visited.add(next_vertex)
            current = next_vertex
        if path[-1] != start:
            path.append(start)
        if len(path) > 1:
            paths.append(path)
            cost = tour_cost(path)
            if cost < best_cost:
                best_cost = cost
                best_path = path[:]
    # Pheromone evaporation
    for edge in list(Pheromone_dict):
        Pheromone_dict[edge] = round((1 - p) * Pheromone_dict[edge], 3)
    # Deposit pheromone along best tours
    for path in paths:
        if len(path) < 2:
            continue
        cost = tour_cost(path)
        reward = 1 / cost if cost > 0 else 1
        for i in range(len(path) - 1):
            edge = edge_key(path[i], path[i + 1])
            Pheromone_dict[edge] = Pheromone_dict.get(edge, 0) + reward
    return Pheromone_dict, best_path

if __name__ == '__main__':
    initializePheromone()
    print("Initial pheromone:", Pheromone)
    for iteration in range(max_iterations):
        Pheromone, best_path = AOC(Pheromone)
        if iteration % 10 == 0 or iteration == max_iterations - 1:
            print(f"Iteration {iteration + 1}")
            print(f"  Best path: {best_path}")
            print(f"  Pheromone: {Pheromone}")
            print()
    my_path = ['A', 'D', 'C', 'B', 'A']
    my_cost = tour_cost(my_path)
    best_cost = tour_cost(best_path) if best_path else None
    print("My path:", my_path)
    print("My path cost:", my_cost)
    print("Best path found by ACO:", best_path)
    print("Best path cost:", best_cost)
    print("Comparison: my cost =", my_cost, "| best cost =", best_cost)
    if best_cost is not None:
        print(f"Difference = {my_cost - best_cost} units more than the best route")
    print("\nFinal pheromone levels:", Pheromone)


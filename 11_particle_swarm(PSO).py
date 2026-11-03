# PRACTICAL 11: Particle Swarm Optimization (PSO)
import random
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

def distance(bird, food):
    return ((bird[0] - food[0]) ** 2 + (bird[1] - food[1]) ** 2) ** 0.5
food = [6, -3]
birds = [[random.uniform(-10, 10), random.uniform(-10, 10)] for _ in range(10)]
velocities = [[0, 0] for _ in birds]
personal_best = [bird[:] for bird in birds]
global_best = min(personal_best, key=lambda bird: sum((bird[i] - food[i]) ** 2 for i in range(2)))
previous_position = global_best[:]
previous_distance = distance(previous_position, food)
distance_history = []
for iteration in range(1, 101):
    for bird_index, bird in enumerate(birds):
        for coordinate in range(2):
            r1, r2 = random.random(), random.random()
            velocities[bird_index][coordinate] = (
                0.5 * velocities[bird_index][coordinate]
                + 1.5 * r1 * (personal_best[bird_index][coordinate] - bird[coordinate])
                + 1.5 * r2 * (global_best[coordinate] - bird[coordinate])
            )
            bird[coordinate] += velocities[bird_index][coordinate]

        score = sum((bird[i] - food[i]) ** 2 for i in range(2))
        best_score = sum((personal_best[bird_index][i] - food[i]) ** 2 for i in range(2))
        if score < best_score:
            personal_best[bird_index] = bird[:]
        if score < sum((global_best[i] - food[i]) ** 2 for i in range(2)):
            global_best = bird[:]

    print(
        f"Iteration {iteration}: position = "
        f"{[round(value, 2) for value in global_best]}, "
        f"distance = {distance(global_best, food):.2f}"
    )
    distance_history.append(distance(global_best, food))

    if distance(global_best, food) < 0.01:
        global_best = food[:]
        break
print("Food position:", food)
print("Previous bird position:", [round(value, 2) for value in previous_position])
print("Previous distance:", round(previous_distance, 2))
print("Iterations used:", iteration)
print("Bird reached:", [round(value, 2) for value in global_best])
print("Final distance:", distance(global_best, food))

plt.plot(range(1, iteration + 1), distance_history, marker="o", markersize=3)
plt.axhline(0, color="red", linestyle="--", label="Food position")
plt.xlabel("Iteration")
plt.ylabel("Distance to food")
plt.title("PSO Convergence: Bird Moving Toward Food")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(Path(__file__).with_name("pso_convergence.png"), dpi=150)
plt.show()

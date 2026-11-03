import random
import math
# import csv                      # uncomment for CSV input

# ============================================================
# INPUT DATA  (use ONE of the two options below)
# ============================================================

# ---------- Option 1: Hard-coded dataset (ACTIVE) ----------
data = [
    [85, 90],
    [80, 88],
    [30, 40],
    [35, 45]
]

# ---------- Option 2: CSV dataset (COMMENTED OUT) ----------
# FILE = "data.csv"
# HAS_LABEL = False               # True if the last column is a class label
#
# with open(FILE) as f:
#     rows = list(csv.reader(f))
# rows = rows[1:]                 # skip header row
# if HAS_LABEL:
#     rows = [r[:-1] for r in rows]
# data = [[float(v) for v in r] for r in rows]

# ============================================================
# PARAMETERS
# ============================================================
num_neurons = 2                   # number of clusters
learning_rate = 0.5
epochs = 100
num_features = len(data[0])

# ============================================================
# MIN-MAX NORMALIZATION (scales every column to 0-1)
# ============================================================
original_data = [row[:] for row in data]     # kept for printing results
mins = [min(col) for col in zip(*data)]
maxs = [max(col) for col in zip(*data)]
data = [[(x[i] - mins[i]) / (maxs[i] - mins[i] or 1) for i in range(num_features)]
        for x in data]

# ============================================================
# HELPER: find Best Matching Unit
# ============================================================
def find_bmu(sample):
    distances = [
        math.sqrt(sum((sample[i] - w[i]) ** 2 for i in range(num_features)))
        for w in weights
    ]
    return distances.index(min(distances))

# ============================================================
# INITIALIZE WEIGHTS (from random samples, avoids dead neurons)
# ============================================================
weights = [random.choice(data)[:] for _ in range(num_neurons)]

# ============================================================
# SOM TRAINING (learning rate decays each epoch)
# ============================================================
for epoch in range(epochs):
    lr = learning_rate * (1 - epoch / epochs)

    for sample in data:
        bmu = find_bmu(sample)
        for i in range(num_features):
            weights[bmu][i] += lr * (sample[i] - weights[bmu][i])

    # Print progress only for the first and last few epochs
    if epoch < 3 or epoch >= epochs - 2:
        print("Epoch", epoch + 1)
        print("Updated Weights:", [[round(v, 3) for v in w] for w in weights])

# ============================================================
# FINAL CLASSIFICATION
# ============================================================
print("\nFinal Classification")
for original, sample in zip(original_data, data):
    print(original, "belongs to Neuron", find_bmu(sample))

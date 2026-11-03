# PRACTICAL 3: Unsupervised Neural Networks (Self Organizing Map)
######################################################################
import pandas as pd
import numpy as np
from minisom import MiniSom
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
)  # =========================

# Hyperparameters
# ============================================================
FILE = "Dataset_spine.csv"
# SOM
ROWS = 5
COLS = 5
SIGMA = 1.0
LEARNING_RATE = 0.2
ITERATIONS = 1000
# K-Means
MAX_ITER = 300
INIT = 10
# ============================================================
# Load Dataset
# ============================================================
data = pd.read_csv(FILE)  # Remove empty columns
data = data.dropna(axis=1, how="all")
data = data.loc[:, ~data.columns.str.contains("Unnamed")]
# ============================================================
# Split Data
# ============================================================
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
# ============================================================
# Convert Target
# ============================================================
# Convert any label type into numbers
y, labels = pd.factorize(y)
y = np.array(y, dtype=int)
CLASSES = len(np.unique(y))
if CLASSES < 2:
    print("Dataset has only one class.")
    print("Clustering comparison is not possible.")
    exit()
# ============================================================
# Feature Processing
# ============================================================
# Convert categorical features
X = pd.get_dummies(X)
# Replace infinity values
X = X.replace([np.inf, -np.inf], np.nan)
# Fill missing values
X = SimpleImputer(strategy="mean").fit_transform(X)
# Normalize
X = StandardScaler().fit_transform(X)
# ============================================================
# SOM
# ============================================================
som = MiniSom(
    ROWS, COLS, X.shape[1], sigma=SIGMA, learning_rate=LEARNING_RATE, random_seed=42
)
som.random_weights_init(X)
som.train_random(X, ITERATIONS)
# SOM neuron clustering
weights = som.get_weights()
weights = weights.reshape(ROWS * COLS, X.shape[1])
som_model = KMeans(n_clusters=CLASSES, random_state=42, n_init=INIT)
neuron_labels = som_model.fit_predict(weights)
som_cluster = []
for sample in X:
    r, c = som.winner(sample)
    index = r * COLS + c
    som_cluster.append(neuron_labels[index])
som_cluster = np.array(som_cluster)
# ============================================================
# K-Means # =========================
km = KMeans(n_clusters=CLASSES, max_iter=MAX_ITER, random_state=42, n_init=INIT)
km_cluster = km.fit_predict(X)

# ============================================================
# Cluster Mapping
# ============================================================
def convert(cluster):
    result = np.zeros(len(cluster), dtype=int)
    for c in np.unique(cluster):
        index = np.where(cluster == c)[0]
        values, counts = np.unique(y[index], return_counts=True)
        result[index] = values[np.argmax(counts)]
    return result

som_prediction = convert(som_cluster)
km_prediction = convert(km_cluster)

# ============================================================
# Metrics
# ============================================================
def calculate_metrics(pred):
    cm = confusion_matrix(y, pred)
    accuracy = np.mean(y == pred)
    sensitivity = recall_score(y, pred, average="weighted", zero_division=0)
    precision = precision_score(y, pred, average="weighted", zero_division=0)
    specificity = []
    for i in range(len(cm)):
        TP = cm[i, i]
        FN = cm[i].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - (TP + FN + FP)
        if TN + FP != 0:
            specificity.append(TN / (TN + FP))
    if len(specificity):
        specificity = np.mean(specificity)
    else:
        specificity = 0
    return (accuracy, sensitivity, specificity, precision)

som_result = calculate_metrics(som_prediction)
km_result = calculate_metrics(km_prediction)
# ============================================================
# Output
# ============================================================
print("\n===================================")
print("Number of Clusters :", CLASSES)
print("===================================")
print("\nMetric              SOM       K-Means")
print("-----------------------------------")
names = ["Accuracy", "Sensitivity", "Specificity", "Precision"]
for i in range(4):
    print(f"{names[i]:15}" f"{som_result[i]*100:8.2f}%" f"{km_result[i]*100:10.2f}%")
print("===================================")

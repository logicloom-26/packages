
# %%
######################################################################
# PRACTICAL 1: ANN for Linear Regression
######################################################################
import numpy as np

X = np.array([
    [2, 3],
    [3, 4],
    [4, 5],
    [5, 6],
    [6, 7]
], dtype=float)                         

Y = np.array([
    [8],
    [11],
    [14],
    [17],
    [20]
], dtype=float)

w1=0.5
w2=0.3
b=0.2

lr=0.01
epochs=10000

print("Initial Parameters :")
print("W1 :",w1)
print("W2 :",w2)
print("Bias :",b)
print("--"*50)

for epoch in range(epochs):
    total_error=0
    for i in range(len(X)):
        x1=X[i][0]
        x2=X[i][1]
        target=Y[i][0]

        net=(x1*w1)+(x2*w2)+b

        output=net

        error=target-output

        total_error+=error**2

        dw1=-error*x1
        dw2=-error*x2
        db=-error

        w1=w1-lr*dw1
        w2=w2-lr*dw2
        b=b-lr*db

    mse=total_error/len(X)

    if epoch % 100==0:
        print(f"Epoch {epoch:4d} MSE= {mse:.6f}")
    if mse<0.00001:
        print("Training Completed")
        print("Epoch :",epoch)
        break

print("Final Parameters :")
print("W1 :",round(w1,4))
print("W2 :",round(w2,4))
print("Bias :",round(b,4))
print("--"*50)

for i in range(len(X)):
    x1=X[i][0]
    x2=X[i][1]

    predicted=(x1*w1)+(x2*w2)+b

    print(
        "Input :",X[i],
        "Actual :",Y[i][0],
        "Predicted :",round(predicted,3),
        "Error :",round(Y[i][0]-predicted,5)
        

        )       

# %%
######################################################################
# PRACTICAL 2: Visualize activation functions
######################################################################
import numpy as np
import matplotlib.pyplot as plt
from math import erf


# Input values
x = np.linspace(-5, 5, 500)


# ---------------- Activation Functions ----------------

def binary_step(x):
    return np.where(x >= 0, 1, 0)


def linear(x):
    return x


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def relu(x):
    return np.maximum(0, x)


def leaky_relu(x):
    return np.where(x > 0, x, 0.01*x)


def prelu(x):
    a = 0.1
    return np.where(x > 0, x, a*x)


def elu(x):
    alpha = 1
    return np.where(x > 0, x, alpha*(np.exp(x)-1))


def selu(x):
    alpha = 1.67326
    scale = 1.0507
    return scale*np.where(x > 0, x, alpha*(np.exp(x)-1))


def softplus(x):
    return np.log(1 + np.exp(x))


def softsign(x):
    return x/(1+np.abs(x))


def swish(x):
    return x*sigmoid(x)


def mish(x):
    return x*np.tanh(softplus(x))


def gelu(x):
    return 0.5*x*(1+np.vectorize(erf)(x/np.sqrt(2)))


def hard_sigmoid(x):
    return np.clip(0.2*x+0.5,0,1)


def hard_swish(x):
    return x*np.clip(x+3,0,6)/6


def maxout(x):
    return np.maximum(x,0.5*x)


def softmax(x):
    e = np.exp(x-np.max(x))
    return e/e.sum()


def elu_variant(x):
    return np.where(x>0,x,np.exp(x)-1)


def absolute_linear(x):
    return np.abs(x)



# ---------------- Activation List ----------------

functions = [

(binary_step,"Binary Step","1 if x≥0 else 0"),
(linear,"Linear","f(x)=x"),
(sigmoid,"Sigmoid","1/(1+e^-x)"),
(tanh,"Tanh","tanh(x)"),
(relu,"ReLU","max(0,x)"),
(leaky_relu,"Leaky ReLU","x or 0.01x"),
(prelu,"PReLU","x or ax"),
(elu,"ELU","α(e^x-1)"),
(selu,"SELU","λ(ELU)"),
(softplus,"Softplus","ln(1+e^x)"),
(softsign,"Softsign","x/(1+|x|)"),
(swish,"Swish","x*sigmoid(x)"),
(mish,"Mish","x*tanh(ln(1+e^x))"),
(gelu,"GELU","0.5x(1+erf(x/√2))"),
(hard_sigmoid,"Hard Sigmoid","clip(0.2x+0.5)"),
(hard_swish,"Hard Swish","x*ReLU6(x+3)/6"),
(maxout,"Maxout","max(x,0.5x)"),
(softmax,"Softmax","e^x/Σe^x"),
(elu_variant,"ELU Variant","e^x-1"),
(absolute_linear,"Absolute Linear","|x|")

]



# ---------------- Plotting Function ----------------

def plot_activation(ax, function, name, formula):

    y = function(x)

    ax.plot(x,y)

    ax.set_title(
        name + "\n" + formula,
        fontsize=9
    )

    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")

    ax.axhline(0)
    ax.axvline(0)

    ax.grid(True)



# ---------------- Create Grid ----------------

rows = 5
cols = 4

fig, axes = plt.subplots(
    rows,
    cols,
    figsize=(16,18)
)
axes = axes.flatten()
for ax, (func,name,formula) in zip(axes,functions):
    plot_activation(
        ax,
        func,
        name,
        formula
    )
# Remove empty plots (if any)
for i in range(len(functions),len(axes)):
    axes[i].axis("off")
plt.tight_layout()
plt.show()

# %%
######################################################################
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

# %%
######################################################################
# PRACTICAL 4: Reinforcement Learning (Q-Learning)
######################################################################
import numpy as np
import gymnasium as gym
import time

# ===================================================================
# CHOOSE THE ENVIRONMENT (use only ONE option at a time)
# ===================================================================

# ---------- OPTION 1 (DEFAULT): Standard 4x4 map ----------
env=gym.make(
    "FrozenLake-v1",
    is_slippery=False
    )

# ---------- OPTION 2: Custom 5x5 map ----------
# To use this: remove the '#' from the lines below
# AND put '#' in front of the OPTION 1 lines above.
#
# MAP = [
#     "SFFFF",
#     "FHFHF",
#     "FFFHF",
#     "HFFFF",
#     "FFHFG"
# ]
#
# env=gym.make(
#     "FrozenLake-v1",
#     desc=MAP,
#     is_slippery=False
#     )

n_observations=env.observation_space.n
n_actions=env.action_space.n

print("Observations :",n_observations)
print("Action :",n_actions)

Q_table=np.zeros((n_observations,n_actions))
print("\nInitial Q table")
print(Q_table)

n_episodes=5000
max_iter_episodes=100
lr=0.2
gamma=0.95
exploration_proba=1
exploration_decreasing_decay=0.001
min_exploration_proba=0.01

reward_per_episode=[]

for e in range(n_episodes):
    current_state,_=env.reset()
    done=False
    total_episode_reward=0
    for i in range(max_iter_episodes):
        if np.random.uniform(0,1)<exploration_proba:
            action=env.action_space.sample()
        else:
            action=np.argmax(Q_table[current_state])

        next_state, reward, done, truncated,_=env.step(action)

        if done:
            Q_table[current_state,action]=((1-lr)*Q_table[current_state,action]+lr*reward)
        else:
            Q_table[current_state,action]=((1-lr)*Q_table[current_state,action]+lr*(reward+gamma*np.max(Q_table[next_state])))

        total_episode_reward+=reward

        if done:
            break

        current_state=next_state

    exploration_proba=max(min_exploration_proba,np.exp(-exploration_decreasing_decay*e))

    reward_per_episode.append(total_episode_reward)

print("\nTraining Completed")
print("\nFinal Q table")
print(Q_table)

print("\nMean reward per thousand episodes")
for i in range(n_episodes//1000):
    mean_reward=np.mean(reward_per_episode[i*1000:(i+1)*1000])
    print(f"{i*1000}-{(i+1)*1000}: {mean_reward}")

print("\nOverall mean reward :",np.mean(reward_per_episode))

env.close()

print("\nStarting visualization...")

# ---------- OPTION 1 (DEFAULT): Standard 4x4 map ----------
test_env=gym.make(
    "FrozenLake-v1",
    is_slippery=False,
    render_mode='human'
    )

# ---------- OPTION 2: Custom 5x5 map ----------
# Uncomment this AND comment OPTION 1 above (MAP must be uncommented at the top)
#
# test_env=gym.make(
#     "FrozenLake-v1",
#     desc=MAP,
#     is_slippery=False,
#     render_mode='human'
#     )

state,_=test_env.reset()
done=False
truncated=False
while not (done or truncated):
    action=np.argmax(Q_table[state])

    state, reward, done, truncated,_=test_env.step(action)

    time.sleep(0.5)

if reward==1:
    print("Agent reached the goal")
else:
    print("Agent fell in the hole")

test_env.close()

# %%
######################################################################
# PRACTICAL 5: Reinforcement Learning with SARSA
######################################################################
import numpy as np
import gymnasium as gym
import time

# ---------- OPTION 1 (DEFAULT): Standard 4x4 map ----------
env = gym.make(
    "FrozenLake-v1",
    is_slippery=False
)

# ---------- OPTION 2: Custom 5x5 map ----------
# To use this: remove the '#' from the lines below
# AND put '#' in front of the OPTION 1 lines above.
#
# MAP = [
#     "SFFFF",
#     "FHFHF",
#     "FFFHF",
#     "HFFFF",
#     "FFHFG"
# ]
#
# env = gym.make(
#     "FrozenLake-v1",
#     desc=MAP,
#     is_slippery=False
# )

n_observations = env.observation_space.n
n_actions = env.action_space.n

print(f"Observations: {n_observations}")
print(f"Actions: {n_actions}")

Q_table = np.zeros((n_observations, n_actions))

print("\nInitial Q Table:")
print(Q_table)

n_episodes = 5000
max_iter_episode = 100
learning_rate = 0.2
gamma = 0.95
exploration_proba = 1
exploration_decreasing_decay = 0.001
min_exploration_proba = 0.01

rewards_per_episode = []

def choose_action(state, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    best_actions = np.flatnonzero(Q_table[state] == np.max(Q_table[state]))
    return np.random.choice(best_actions)


# -----------------------------------
# SARSA Training
# -----------------------------------
for e in range(n_episodes):
    current_state, _ = env.reset()
    total_episode_reward = 0

    current_action = choose_action(current_state, exploration_proba)

    for i in range(max_iter_episode):

        next_state, reward, done, truncated, _ = env.step(current_action)

        
        next_action = choose_action(next_state, exploration_proba)

                if done:
            target = reward
        else:
            target = reward + gamma * Q_table[next_state, next_action]

        Q_table[current_state, current_action] = ((1 - learning_rate) * Q_table[current_state, current_action] + learning_rate * target)

        total_episode_reward += reward

        if done:
            break

       
        current_state = next_state
        current_action = next_action

    
    exploration_proba = max(min_exploration_proba, np.exp(-exploration_decreasing_decay * e))

    rewards_per_episode.append(total_episode_reward)


print("\nTraining Completed!")

print("\nFinal Q Table:")
print(Q_table)

print("\nMean reward per thousand episodes")

for i in range(n_episodes // 1000):
    mean_reward = np.mean(rewards_per_episode[i * 1000:(i + 1) * 1000])
    print(f"{i * 1000} - {(i + 1) * 1000}: {mean_reward}")

print("\nOverall Mean Reward:", np.mean(rewards_per_episode))

env.close()


print("\nStarting Visualization...")

# ---------- OPTION 1 (DEFAULT): Standard 4x4 map ----------
test_env = gym.make(
    "FrozenLake-v1",
    is_slippery=False,
    render_mode="human"
)

# ---------- OPTION 2: Custom 5x5 map ----------
# Uncomment this AND comment OPTION 1 above (MAP must be uncommented at the top)
#
# test_env = gym.make(
#     "FrozenLake-v1",
#     desc=MAP,
#     is_slippery=False,
#     render_mode="human"
# )

state, _ = test_env.reset()
done = False
truncated = False

while not (done or truncated):

    action = np.argmax(Q_table[state])

    state, reward, done, truncated, _ = test_env.step(action)

    time.sleep(0.5)

if reward == 1:
    print("Agent reached the goal!")
else:
    print("Agent fell into the hole!")

test_env.close()



######################################################################
# PRACTICAL 6: Reinforcement Learning with Neural Networks (Deep Q-Network)
######################################################################
# ============================================================
# Reinforcement Learning with Neural Networks (Deep Q-Network)
# Environment : FrozenLake 4x4 (is_slippery=False)
# Idea        : Instead of a Q-table, a small neural network
#               takes the STATE and predicts the Q-VALUE of
#               each of the 4 actions.
# ============================================================
import numpy as np
import gymnasium as gym
import random
import time
from collections import deque

# ---------- OPTION 1 (DEFAULT): Standard 4x4 map ----------
env = gym.make(
    "FrozenLake-v1",
    is_slippery=False
)

# ---------- OPTION 2: Custom 5x5 map ----------
# To use this: remove the '#' from the lines below
# AND put '#' in front of the OPTION 1 lines above.
#
# MAP = [
#     "SFFFF",
#     "FHFHF",
#     "FFFHF",
#     "HFFFF",
#     "FFHFG"
# ]
#
# env = gym.make(
#     "FrozenLake-v1",
#     desc=MAP,
#     is_slippery=False
# )

n_observations = env.observation_space.n
n_actions = env.action_space.n

print(f"Observations: {n_observations}")
print(f"Actions: {n_actions}")

def one_hot(state):
    v = np.zeros(n_observations)
    v[state] = 1.0
    return v

class NeuralNetwork:
    def __init__(self, n_in, n_hidden, n_out):
        self.W1 = np.random.randn(n_in, n_hidden) * 0.1
        self.b1 = np.zeros(n_hidden)
        self.W2 = np.random.randn(n_hidden, n_out) * 0.1
        self.b2 = np.zeros(n_out)

    def forward(self, x):
        self.h = np.maximum(0, x @ self.W1 + self.b1)
        return self.h @ self.W2 + self.b2

    def train_step(self, x, action, target, learning_rate):
        q = self.forward(x)
        d_out = np.zeros_like(q)
        d_out[action] = q[action] - target

        d_h = (self.W2 @ d_out) * (self.h > 0)

        self.W2 -= learning_rate * np.outer(self.h, d_out)
        self.b2 -= learning_rate * d_out
        self.W1 -= learning_rate * np.outer(x, d_h)
        self.b1 -= learning_rate * d_h

    def copy_from(self, other):
        self.W1, self.b1 = other.W1.copy(), other.b1.copy()
        self.W2, self.b2 = other.W2.copy(), other.b2.copy()

q_net = NeuralNetwork(n_observations, 32, n_actions)
target_net = NeuralNetwork(n_observations, 32, n_actions)
target_net.copy_from(q_net)

n_episodes = 2000
max_iter_episode = 100
learning_rate = 0.01
gamma = 0.95
exploration_proba = 1
exploration_decreasing_decay = 0.003
min_exploration_proba = 0.05
batch_size = 32
target_update = 20

memory = deque(maxlen=2000)
rewards_per_episode = []

# -----------------------------------
# Deep Q Learning Training
# -----------------------------------
for e in range(n_episodes):
    current_state, _ = env.reset()
    total_episode_reward = 0

    for i in range(max_iter_episode):

        if np.random.uniform(0, 1) < exploration_proba:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_net.forward(one_hot(current_state)))

        next_state, reward, done, truncated, _ = env.step(action)

        memory.append((current_state, action, reward, next_state, done))

        if len(memory) >= batch_size:
            batch = random.sample(memory, batch_size)
            for s, a, r, s2, d in batch:
                if d:
                    target = r
                else:
                    target = r + gamma * np.max(target_net.forward(one_hot(s2)))
                q_net.train_step(one_hot(s), a, target, learning_rate)

        total_episode_reward += reward

        if done:
            break

        current_state = next_state

    exploration_proba = max(min_exploration_proba, np.exp(-exploration_decreasing_decay * e))

    if (e + 1) % target_update == 0:
        target_net.copy_from(q_net)

    rewards_per_episode.append(total_episode_reward)


print("\nTraining Completed!")

print("\nFinal Q Values:")
for s in range(n_observations):
    print(f"State {s}:", np.round(q_net.forward(one_hot(s)), 3))

print("\nMean reward per thousand episodes")

for i in range(n_episodes // 1000):
    mean_reward = np.mean(rewards_per_episode[i * 1000:(i + 1) * 1000])
    print(f"{i * 1000} - {(i + 1) * 1000}: {mean_reward}")

print("\nOverall Mean Reward:", np.mean(rewards_per_episode))

env.close()


print("\nStarting Visualization...")

# ---------- OPTION 1 (DEFAULT): Standard 4x4 map ----------
test_env = gym.make(
    "FrozenLake-v1",
    is_slippery=False,
    render_mode="human"
)

# ---------- OPTION 2: Custom 5x5 map ----------
# Uncomment this AND comment OPTION 1 above (MAP must be uncommented at the top)
#
# test_env = gym.make(
#     "FrozenLake-v1",
#     desc=MAP,
#     is_slippery=False,
#     render_mode="human"
# )

state, _ = test_env.reset()
done = False
truncated = False

time.sleep(2)

while not (done or truncated):

    action = np.argmax(q_net.forward(one_hot(state)))

    state, reward, done, truncated, _ = test_env.step(action)

    time.sleep(1)

time.sleep(3)

if reward == 1:
    print("Agent reached the goal!")
else:
    print("Agent fell into the hole!")

test_env.close()


# %%
######################################################################
# PRACTICAL 7: Radial Basis Function Neural Network (XOR)
######################################################################
# ============================================================
# Radial Basis Function Neural Network (RBFNN) - XOR Problem
# Structure : 2 inputs -> 4 Gaussian hidden neurons -> 1 output
# Idea      : Each hidden neuron measures how CLOSE the input is
#             to its centre. XOR is not linearly separable, but
#             after this transformation it becomes separable.
# ============================================================
import numpy as np

# ==========================================
# Non-linear XOR Input Data
# ==========================================
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# ==========================================
# RBF Network Parameters
# ==========================================
input_neurons = 2
hidden_neurons = 4
output_neurons = 1
learning_rate = 0.5
sigma = 1.0
epochs = 1000

# ==========================================
# Hidden Layer Centers
# ==========================================
centers = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# ==========================================
# Initial Manual Weights
# w1 w2 w3 w4
# ==========================================
W = np.array([
    -0.8,  # w1
    0.9,   # w2
    0.9,   # w3
    -0.8   # w4
])

bias = 0.1

print("Initial Weights")
print("w1 =", W[0])
print("w2 =", W[1])
print("w3 =", W[2])
print("w4 =", W[3])
print("Initial Bias =", bias)


# ==========================================
# Gaussian RBF Activation Function
# ==========================================
def gaussian_rbf(x, center, sigma):
    distance = np.linalg.norm(x - center)
    activation = np.exp(
        -(distance ** 2) /
        (2 * sigma ** 2)
    )
    return activation


# ==========================================
# Function to generate XOR target
# No Y array used
# ==========================================
def xor_target(x):
    if np.array_equal(x, [0, 0]):
        return 0
    elif np.array_equal(x, [1, 1]):
        return 0
    else:
        return 1


# ==========================================
# Training using Gradient Descent
# ==========================================
for epoch in range(epochs):
    total_error = 0

    for x in X:
        # Generate target internally
        target = xor_target(x)

        # --------------------------
        # Hidden Layer
        # --------------------------
        H = []
        for c in centers:
            H.append(
                gaussian_rbf(
                    x,
                    c,
                    sigma
                )
            )
        H = np.array(H)

        # --------------------------
        # Output Layer
        # y=w1h1+w2h2+w3h3+w4h4+b
        # --------------------------
        y = np.dot(W, H) + bias

        # --------------------------
        # Error
        # --------------------------
        error = target - y
        total_error += error ** 2

        # --------------------------
        # Gradient Descent Update
        # --------------------------
        W = W + learning_rate * error * H
        bias = bias + learning_rate * error

    # Print every 100 epochs
    if epoch % 100 == 0:
        print("\nEpoch:", epoch)
        print(
            "Error:",
            round(total_error, 4)
        )
        print("Weights:")
        print("w1 =", round(W[0], 4))
        print("w2 =", round(W[1], 4))
        print("w3 =", round(W[2], 4))
        print("w4 =", round(W[3], 4))
        print(
            "Bias =",
            round(bias, 4)
        )

# ==========================================
# Final Trained Weights
# ==========================================
print("\n==============================")
print("Final Trained Weights")
print("==============================")
print("w1 =", round(W[0], 4))
print("w2 =", round(W[1], 4))
print("w3 =", round(W[2], 4))
print("w4 =", round(W[3], 4))
print(
    "Final Bias =",
    round(bias, 4)
)

# ==========================================
# Testing
# ==========================================
print("\nTesting\n")

for x in X:
    # Hidden Layer
    H = []
    for c in centers:
        H.append(
            gaussian_rbf(
                x,
                c,
                sigma
            )
        )
    H = np.array(H)

    # Output
    y = np.dot(W, H) + bias

    # Classification
    prediction = 1 if y >= 0.5 else 0

    print(
        "Input:",
        x,
        "Raw Output:",
        round(y, 2),
        "Class Output:",
        f"{prediction:.2f}"
    )

# %%
######################################################################
# PRACTICAL 8: Genetic Algorithms - I
######################################################################
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

# %%
######################################################################
# PRACTICAL 9: Genetic Algorithms - II
# Part 1: Tournament vs Rank Selection (INDIA)
######################################################################
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
# %%
######################################################################
# PRACTICAL 9: Genetic Algorithms - II
# Part 2: Compare all selections (INDIA)
######################################################################
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

######################################################################
# PRACTICAL 9: Genetic Algorithms - II
# Part 3: Knapsack Problem
######################################################################
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

# %%
######################################################################
# PRACTICAL 11: Particle Swarm Optimization (PSO)
######################################################################
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

# %%
######################################################################
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

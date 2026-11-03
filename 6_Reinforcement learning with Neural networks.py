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

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

import BombermanEnv
import QlearningAgent as Qlearning

import matplotlib.pyplot as plt
import os
import numpy as np

def running_average(x, N):
    cumsum = np.cumsum(np.insert(np.array(x), 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N

file_str = "C:/Users/Harunomi/Desktop/Programas Pencas/Aprendizaje-por-Refuerzo_2-2023/Lab2/cajas.txt"
env = BombermanEnv.BombermanEnv(5,5,1,1,1,file_str,'rgb_array')

# Parameters
gamma = 0.9
epsilon = 1.0
decay_epsilon = 1e-5
alpha = 0.1
nb_episodes = 20000

agent = Qlearning.QLearningAgent(env, gamma, epsilon, decay_epsilon, alpha)

returns, steps = agent.train(nb_episodes)

# Plot training returns
plt.figure(figsize=(15, 6))
plt.subplot(121)
plt.plot(returns)
plt.plot(running_average(returns, 1000))
plt.xlabel("Episodes")
plt.ylabel("Returns")
plt.subplot(122)
plt.plot(steps)
plt.plot(running_average(steps, 1000))
plt.xlabel("Episodes")
plt.ylabel("steps")
plt.show()

# Test the agent for 1000 episodes
test_returns = []
test_steps = []
for episode in range(1000):
    return_episode, nb_steps = agent.test()
    test_returns.append(return_episode)
    test_steps.append(nb_steps)
print("Test performance", np.mean(test_returns))

plt.figure(figsize=(15, 6))
plt.subplot(121)
plt.hist(test_returns)
plt.xlabel("Returns")
plt.subplot(122)
plt.hist(test_steps)
plt.xlabel("Number of steps")
plt.show()
import BombermanEnv
import Qlearning

import matplotlib.pyplot as plt
import os

file_str = "C:/Users/Harunomi/Desktop/Programas Pencas/Aprendizaje-por-Refuerzo_2-2023/Lab2/cajas.txt"
env = BombermanEnv.BombermanEnv(5,5,1,1,1,file_str,'rgb_array')

# Parameters
gamma = 0.9
epsilon = 1.0
decay_epsilon = 1e-5
alpha = 0.1
nb_episodes = 20000

agent = Qlearning.MonteCarloAgent(env, gamma, epsilon, decay_epsilon, alpha)

returns, steps = agent.train(20000, recorder=True)

plt.figure(figsize=(10, 6))
plt.subplot(121)
plt.plot(returns)
plt.xlabel("Episodes")
plt.ylabel("Returns")
plt.subplot(122)
plt.plot(steps)
plt.xlabel("Episodes")
plt.ylabel("steps")
plt.show()
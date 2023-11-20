import BombermanEnv
import MonteCarloAgent

import matplotlib.pyplot as plt
import os

file_str = "C:/Users/Harunomi/Desktop/Programas Pencas/Aprendizaje-por-Refuerzo_2-2023/Lab2/cajas.txt"
env = BombermanEnv.BombermanEnv(5,5,1,1,1,file_str,'rgb_array')

agent = MonteCarloAgent.MonteCarloAgent(env, 0.9, 0.1, 0.1)

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
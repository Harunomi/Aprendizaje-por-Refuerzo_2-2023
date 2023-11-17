import BombermanEnv
import MonteCarloAgent


env = BombermanEnv.BombermanEnv(5,5,10,2,2,'','human')

agent = MonteCarloAgent.MonteCarloAgent(env, 0.9, 0.1, 0.1)

returns, steps = agent.train(1000, recorder=True)

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
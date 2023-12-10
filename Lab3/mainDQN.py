from matplotlib import pyplot as plt
import numpy as np
import BombermanEnv
import DQNAgent

def running_average(x, N):
    kernel = np.ones(N) / N
    return np.convolve(x, kernel, mode='same')

# Parameters
nb_episodes = 150
batch_size = 32

epsilon = 1.0
epsilon_decay = 0.0005

gamma = 0.99

learning_rate = 0.005
buffer_limit = 5000
target_update_period = 100
training_update_period = 4

file_str = 'Lab3/cajas.txt'

# Create the environment
env = BombermanEnv.BombermanEnv(5,5,1,0,1,file_str,'rgb_array')

# Create the agent
agent = DQNAgent.DQNAgent(env, learning_rate, epsilon, epsilon_decay, gamma, batch_size, target_update_period, training_update_period, buffer_limit)

# Train the agent
returns, losses = agent.train(nb_episodes)

# Plot the returns
plt.figure(figsize=(10, 6))
plt.plot(returns)
plt.plot(running_average(returns, 10))
plt.xlabel("Episodes")
plt.ylabel("Returns")

# Plot the losses
plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.xlabel("Episodes")
plt.ylabel("Training loss")
plt.show()
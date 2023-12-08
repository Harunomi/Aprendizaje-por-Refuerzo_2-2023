from collections import deque
import numpy as np


class ReplayBuffer:
    def __init__(self, max_capacity):
        self.max_capacity = max_capacity

        # deques, uno por cada elemento
        self.states = deque(maxlen=max_capacity)
        self.actions = deque(maxlen=max_capacity)
        self.rewards = deque(maxlen=max_capacity)
        self.next_states = deque(maxlen=max_capacity)
        self.dones = deque(maxlen=max_capacity)

    def append(self, state, action, reward, next_state, done):
        # Almacenar una transicion
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)

    def sample(self, batch_size):
        if len(self.states) < 2 * batch_size:
            return []

        indices = sorted(np.random.choice(np.arange(len(self.states)), batch_size, replace=False))

        return [
            np.stack([np.array(self.states[i]) for i in indices]),
            np.stack([np.array(self.actions[i]) for i in indices]),
            np.stack([np.array(self.rewards[i]) for i in indices]),
            np.stack([np.array(self.next_states[i]) for i in indices]),
            np.stack([np.array(self.dones[i]) for i in indices])
        ]
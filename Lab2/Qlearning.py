class QLearningAgent:
    """
    Q-learning agent.
    """

    def __init__(self, env, gamma, epsilon, decay_epsilon, alpha):
        """
        :param env: gym-like environment
        :param gamma: discount factor
        :param epsilon: exploration parameter
        :param decay_epsilon: exploration decay parameter
        :param alpha: learning rate
        """
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.decay_epsilon = decay_epsilon
        self.alpha = alpha

        # Q_table
        self.Q = np.zeros([self.env.observation_space.n, self.env.action_space.n])

    def act(self, state):
        "Returns an action using epsilon-greedy action selection."

        action = rng.choice(np.where(self.Q[state, :] == self.Q[state, :].max())[0])

        if rng.random() < self.epsilon:
            action = self.env.action_space.sample()

        return action

    def update(self, state, action, reward, next_state, done):
        "Updates the agent using a single transition."

        # Bellman target
        target = reward

        if not done:
            target += self.gamma * self.Q[next_state, :].max()

        # Update the Q-value
        self.Q[state, action] += self.alpha * (target - self.Q[state, action])

        # Decay epsilon
        self.epsilon = self.epsilon * (1 - self.decay_epsilon)


    def train(self, nb_episodes, recorder=None):
        "Runs the agent on the environment for nb_episodes. Returns the list of obtained returns."

        # Returns
        returns = []
        steps = []

        # Fixed number of episodes
        for episode in range(nb_episodes):

            # Reset
            state, info = self.env.reset()
            done = False
            nb_steps = 0

            # Store rewards
            return_episode = 0.0

            # Sample the episode
            while not done:

                # Select an action
                action = self.act(state)

                # Perform the action
                next_state, reward, terminal, truncated, info = self.env.step(action)

                # Learn from the transition
                self.update(state, action, reward, next_state, done)

                # Go in the next state
                state = next_state

                # Increment time
                nb_steps += 1
                return_episode += reward

                # End of the episode
                done = terminal or truncated

            # Record at the end of the episode
            if recorder is not None and episode == nb_episodes -1:
                recorder.record(self.env.render())

            # Store info
            returns.append(return_episode)
            steps.append(nb_steps)


        return returns, steps
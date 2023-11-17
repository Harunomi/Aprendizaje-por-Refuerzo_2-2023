import numpy as np
rng = np.random.default_rng()

class MonteCarloAgent:
    '''
    Monte-Carlo agent.
    '''

    def __init__(self, env, gamma, epsilon, alpha):
        '''
        Constructor.

        @param env: the environment
        @param gamma: the discount factor
        @param epsilon: the probability to explore
        @param alpha: the learning rate
        '''
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha


        # Q_table
        self.Q = np.zeros([self.env.total_states(), self.env.action_space.n])
        
        print(self.Q)


    def act(self, state):

        max_indices = np.where(self.Q[state, :] == self.Q[state, :].max())[0]
        action = rng.choice(max_indices)

        if rng.uniform() < self.epsilon:
            action = rng.choice(self.env.action_space.n)
        
        return action
    
    def update(self, episode):

        return_episode = 0.0

        # iterate backwards over the episode

        for state, action, reward in reversed(episode):

            return_episode = reward + self.gamma * return_episode

            self.Q[state, action] += self.alpha * (return_episode - self.Q[state, action])

    def train(self, nb_episode, recorder=False):
        ''''
        Runs the agent on the environment
        '''

        for episode in range(nb_episode):
            
            # reset
            state, info = self.env.reset()
            return_episode = 0.0
            nb_steps = 0
            done = False

            transition = []

            while not done:
                # act
                action = self.act(state)

                # step
                next_state, reward, info = self.env.step(action)

                # update
                transition.append((state, action, reward))

                state = next_state

                done = terminal or truncated
                nb_steps += 1

                return_episode += reward 

            self.update(transition)

            # Store info
            returns.append(return_episode)
            steps.append(nb_steps)
        
        return returns, steps

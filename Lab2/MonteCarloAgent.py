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
        self.Q = {}
        

    def act(self, state):
        # Convertir la lista a una tupla usando hash para hacerla hashable
        state_key = hash(str(state))
        
        # Acceder al valor Q correspondiente al estado
        q_values = self.Q.get(state_key, np.zeros(self.env.action_space.n))
        action = np.argmax(q_values)

        if rng.uniform() < self.epsilon:
            action = rng.choice(self.env.action_space.n)
        
        action_to_names = {
            0: 'RIGHT',
            1: 'DOWN',
            2: 'LEFT',
            3: 'UP',
            4: 'BOMB',
            5: 'WAIT',
        }

        #print("Action: ", action_to_names[action])
        
        return action
    
    def update(self, episode):

        return_episode = 0.0

        # iterate backwards over the episode

        for state, action, reward in reversed(episode):
            
            state_key = hash(str(state))


            return_episode = reward + self.gamma * return_episode
            q_values = self.Q.get(state_key, np.zeros(self.env.action_space.n))
            q_values[action] = q_values[action] + self.alpha * (return_episode - q_values[action])
            self.Q[state_key] = q_values


    def train(self, nb_episode, recorder=False):
        ''''
        Runs the agent on the environment
        '''
        returns = []
        steps = []

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
                next_state, reward,terminal, truncated, info = self.env.step(action)

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
        
        return returns,steps

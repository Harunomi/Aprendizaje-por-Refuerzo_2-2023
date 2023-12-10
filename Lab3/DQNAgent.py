import numpy as np
rng = np.random.default_rng()
import matplotlib.pyplot as plt
import os
from IPython.display import clear_output
from collections import deque
import tensorflow as tf
import logging
import ReplayBuffer



class DQNAgent:

    def __init__(self, env, learning_rate, epsilon, epsilon_decay, gamma, batch_size, target_update_period, training_update_period, buffer_limit):
        self.env = env

        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_period = target_update_period
        self.training_update_period = training_update_period

        # Create the Q-network and the target network
        tf.keras.backend.clear_session() # start by deleting all existing models to be gentle on the RAM
        self.model = self.build_model(self.learning_rate)
        self.target_model = self.build_model(self.learning_rate)
        self.target_model.set_weights(self.model.get_weights())

        # Create the replay memory
        self.buffer = ReplayBuffer.ReplayBuffer(buffer_limit)

    def build_model(self, learning_rate):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(64, input_shape=(self.env.observation_space.shape[0],), activation='relu'))
        model.add(tf.keras.layers.Dense(self.env.action_space.n, activation='linear'))  # Utiliza env.action_space.n

        # Compila el modelo con el optimizador y la función de pérdida apropiados
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss='mse')

        return model

    def act(self, state):

        # epsilon-greedy
        if np.random.rand() < self.epsilon: # Random selection
            action = self.env.action_space.sample()
        else: # Use the Q-network to get the greedy action
            action = self.model.predict(state.reshape((1, self.env.observation_space.shape[0])), verbose=0)[0].argmax()

        # Decay epsilon
        self.epsilon *= 1 - self.epsilon_decay
        self.epsilon = max(0.05, self.epsilon)

        return action

    def update(self, batch):

        # Get the minibatch
        states, actions, rewards, next_states, dones = batch

        # Predict the Q-values in the current state
        targets = np.array(self.model.predict_on_batch(states))

        # Predict the Q-values in the next state using the target model
        next_Q_value = np.array(self.target_model.predict_on_batch(next_states)).max(axis=1)

        # Terminal states have a value of 0
        next_Q_value[dones] = 0.0

        # Compute the target
        for i in range(self.batch_size):
            targets[i, actions[i]] = rewards[i] + self.gamma * next_Q_value[i]

        # Train the model on the minibatch
        history = self.model.fit(states, targets, epochs=1, batch_size=self.batch_size, verbose=0)

        return history.history['loss'][0]

    def train(self, nb_episodes):

        steps = 0
        returns = []
        losses = []

        for episode in range(nb_episodes):

            # Reset
            state, info = self.env.reset()
            done = False
            steps_episode = 0
            return_episode = 0

            loss_episode = []

            # Sample the episode
            while not done:

                # Select an action
                action = self.act(state)

                # Perform the action
                next_state, reward, terminal, truncated, info = self.env.step(action)

                # End of the episode
                done = terminal or truncated

                # Store the transition
                self.buffer.append(state, action, reward, next_state, done)

                # Sample a minibatch
                batch = self.buffer.sample(self.batch_size)

                # Train the NN on the minibatch
                if len(batch) > 0 and steps % self.training_update_period == 0:
                    loss = self.update(batch)
                    loss_episode.append(loss)

                # Update the target model
                if steps > self.target_update_period and steps % self.target_update_period == 0:
                    self.target_model.set_weights(self.model.get_weights())

                # Go in the next state
                state = next_state

                # Increment time
                steps += 1
                steps_episode += 1
                return_episode += reward

                if done:
                    break

            # Store info
            returns.append(return_episode)
            losses.append(np.mean(loss_episode))

            # Print info
            clear_output(wait=True)
            print('Episode', episode+1)
            print(' total steps:', steps)
            print(' length of the episode:', steps_episode)
            print(' return of the episode:', return_episode)
            print(' current loss:', np.mean(loss_episode))
            print(' epsilon:', self.epsilon)

        return returns, losses

    def test(self):

        old_epsilon = self.epsilon
        self.epsilon = 0.0

        state, info = self.env.reset()
        nb_steps = 0
        done = False

        while not done:
            action = self.act(state)
            next_state, reward, terminal, truncated, info = self.env.step(action)
            done = terminal or truncated
            state = next_state
            nb_steps += 1

        self.epsilon = old_epsilon
        return nb_steps
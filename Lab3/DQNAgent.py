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
        self.model = self.create_model(self.env, self.learning_rate)
        self.target_model = self.create_model(self.env, self.learning_rate)
        self.target_model.set_weights(self.model.get_weights())

        # Create the replay memory
        self.buffer = ReplayBuffer.ReplayBuffer(buffer_limit)

    def create_model(env, lr):
        model = tf.keras.models.Sequential()

        # Cambia la forma de entrada a (None, 6) después de aplanar todas las entradas
        model.add(tf.keras.layers.Flatten(input_shape=(3, 2)))

        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(env.action_space.n, activation='linear'))

        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=lr))

        print(model.summary())

        return model

    def create_model(self, env, lr):
        model = tf.keras.models.Sequential()

        # Ajustar la forma de entrada
        model.add(tf.keras.layers.Input(shape=(env.observation_space.shape[0],), name="flattened_state"))

        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(env.action_space.n, activation='linear'))

        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=lr))

        print(model.summary())

        return model
    
    def update(self, batch):
        # Obtén la minibatch
        states, actions, rewards, next_states, dones = batch

        # Aplana los estados de la minibatch
        states = states.reshape((-1, 6))

        # Predict the Q-values in the current state
        print("               ESTADOS:",states)
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
                print(state)
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
            if episode % 1000 == 0:
                clear_output(wait=True)
                print('Episode', episode+1)
                print(' total steps:', steps)
                print(' length of the episode:', steps_episode)
                print(' return of the episode:', return_episode)
                print(' current loss:', np.mean(loss_episode))
                print(' epsilon:', self.epsilon)

        return returns, losses
    
    def test(self, render=True):

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
import os

import gym
import random
import time
import copy

import numpy as np
import tensorflow as tf

from collections import deque
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.models import Sequential
from keras.optimizers import RMSprop, Adam
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras import backend as K

#EPISODES = 50000
EPISODES = 1000

STORING_PATH = "/home/msabatelli/"


class DeepDQVAgent:
    def __init__(self, action_size, final_v_activation):

        self.render = True
        self.load_model = False
        self.state_size = (84, 84, 4)
        self.action_size = action_size
        self.epsilon = .5
        self.epsilon_start, self.epsilon_end = 1, 0.1
        self.exploration_steps = 1000000.
        self.epsilon_decay_step = (
	self.epsilon_start - self.epsilon_end) / self.exploration_steps

        self.update_target_rate = 10.

        self.tau = 1
        self.tau_decay = .1
        self.minimum_tau = 0.001
        self.clip = (-1, 1)

        self.learning_rate = 0.001
        self.discount_factor = 0.99

        self.memory = deque(maxlen=400000)
        self.no_op_steps = 30

        self.batch_size = 32
        self.train_start = 50000

        self.q_model = self.build_q_model()

        self.final_v_activation = final_v_activation

        self.value_model = self.build_v_model()
        self.target_value_model = self.build_v_model()

        self.optimizer = self.optimizer()

    def update_v_target_model(self):
        self.target_value_model.set_weights(self.value_model.get_weights())

    def build_q_model(self):
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides=(4, 4),
                         activation='relu', input_shape=self.state_size))
        model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_size))
        #model.compile(loss='mse', optimizer=RMSprop(lr=self.learning_rate, rho=0.95, epsilon=0.01))

        return model

    def build_v_model(self):
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides=(4, 4),
                         activation='relu', input_shape=self.state_size))
        model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation=self.final_v_activation))
        model.add(Dense(1))
        model.compile(loss='mse', optimizer=RMSprop(
            lr=self.learning_rate, rho=0.95, epsilon=0.01))

        return model

    def optimizer(self):

        a = K.placeholder(shape=(None,), dtype='int32')
        y = K.placeholder(shape=(None,), dtype='float32')

        py_x = self.q_model.output

        a_one_hot = K.one_hot(a, self.action_size)
        q_value = K.sum(py_x * a_one_hot, axis=1)

        error = K.abs(y - q_value)

        quadratic_part = K.clip(error, 0.0, 1.0)
        linear_part = error - quadratic_part

        loss = K.mean(0.5 * K.square(quadratic_part) + linear_part)

        optimizer = RMSprop(lr=0.00025, epsilon=0.01)
        updates = optimizer.get_updates(
            self.q_model.trainable_weights, [], loss)
        train = K.function([self.q_model.input, a, y], [loss], updates=updates)

        return train

    def save_model(self, name):
        self.model.save_weights(name)

    def get_action(self, history):

        history = np.float32(history / 255.0)

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        else:
            q_value = self.q_model.predict(history)

            return np.argmax(q_value[0])

    def get_Boltzman_action(self, history):

        history = np.float32(history / 255.0)

        # Maxwell Boltzman policy

        q_values = self.q_model.predict(history)
        nb_actions = q_values.shape[0]

        if np.random.uniform() < self.epsilon:
            exp_values = np.exp(
                np.clip(q_values / self.tau, self.clip[0], self.clip[1]))
            probs = exp_values / np.sum(exp_values)
            action = np.random.choice(3, p=probs[0])
        else:
            action = np.argmax(q_values)

        return action

    def store_replay_memory(self, history, action, reward, next_history, dead):
        self.memory.append((history, action, reward, next_history, dead))

    def train_replay(self):

        if len(self.memory) < self.train_start:
            return

        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay_step

        mini_batch = random.sample(self.memory, self.batch_size)

        history = np.zeros((self.batch_size, self.state_size[0],
                            self.state_size[1], self.state_size[2]))
        next_history = np.zeros((self.batch_size, self.state_size[0],
                                 self.state_size[1], self.state_size[2]))

        v_target = np.zeros((self.batch_size,))

        action, reward, dead = [], [], []

        for i in range(self.batch_size):
            history[i] = np.float32(mini_batch[i][0] / 255.)
            next_history[i] = np.float32(mini_batch[i][3] / 255.)
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            dead.append(mini_batch[i][4])

        q_target = self.q_model.predict(history)

        v_target_value = self.target_value_model.predict(next_history)

        q_targets = list()

        for i in range(self.batch_size):

            if dead[i]:
                v_target[i] = reward[i]
                q_target[i][action[i]] = reward[i]

            else:
                v_target[i] = reward[i] + \
                    self.discount_factor * v_target_value[i]
                q_target[i][action[i]] = reward[i] + \
                    self.discount_factor * v_target_value[i]

            q_targets.append(q_target[i][action[i]])

        loss = self.optimizer([history, action, q_targets])

        self.value_model.fit(history, v_target, epochs=1, verbose=0)

        # End of updating the DCNNs in experience replay batch

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def pre_processing(self, observe):
        processed_observe = np.uint8(
            resize(rgb2gray(observe), (84, 84), mode='constant') * 255)

        return processed_observe


if __name__ == "__main__":

    import tensorflow as tf

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--agent', type=str)
    parser.add_argument('--game', type=str)
    parser.add_argument('--exploration', type=str)
    parser.add_argument('--final_v_activation', type=str)

    args = parser.parse_args()

    agent = args.agent
    game = args.game
    exploration = args.exploration
    activation = args.final_v_activation

    env = gym.make(game)
    agent = DeepDQVAgent(action_size=3, final_v_activation=activation)

    global_step = 0
    scores, episodes = [], []

    best_score = 0

    for e in range(EPISODES):

        done = False
        dead = False

        step, score, start_life = 0, 0, 5
        observe = env.reset()

        for _ in range(random.randint(1, agent.no_op_steps)):
            observe, _, _, _ = env.step(1)

        state = agent.pre_processing(observe)
        history = np.stack((state, state, state, state), axis=2)
        # We consider a state as 4 frames in a row
        history = np.reshape([history], (1, 84, 84, 4))

        # (1) State is initialized in history

        while not done:

            if agent.render:
                env.render()

            global_step += 1
            step += 1

            # (2) For S_t we take a by following exploration policy

            if exploration == "e-greedy":
                action = agent.get_action(history)

            elif exploration == "Maxwell-Boltzman":
                action = agent.get_Boltzman_action(history)

            if action == 0:
                real_action = 1
            elif action == 1:
                real_action = 2
            else:
                real_action = 3

            observe, reward, done, info = env.step(real_action)

            next_state = agent.pre_processing(observe)
            next_state = np.reshape([next_state], (1, 84, 84, 1))
            next_history = np.append(next_state, history[:, :, :, :3], axis=3)

            # (3) We store this set of states in the experience replay buffer

            agent.store_replay_memory(
                history, action, reward, next_history, dead)

            # (4) we train the DCNNs on data in the experience replay

            agent.train_replay()

            # (5) we select a_t+1 with usual policy

            if exploration == "e-greedy":
                next_action = agent.get_action(next_history)

            elif exploration == "Maxwell-Boltzman":
                next_action = agent.get_Boltzman_action(next_history)

            if start_life > info['ale.lives']:
                dead = True
                start_life = info['ale.lives']

            reward = np.clip(reward, -1., 1.)

            # (6) We update the V-target model since it is the one computing the TD error

            if global_step % agent.update_target_rate == 0:
                agent.update_v_target_model()

            if reward == +1:
                score += reward

            if dead:
                dead = False
            else:
                history = next_history

            scores.append(score)

        if score > best_score:
            print("New best score for episode: ", score)
            best_score = score

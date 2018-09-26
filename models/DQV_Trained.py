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

EPISODES = 2000

GAME = "Boxing-v4"
V_WEIGHTS = "./boxing/V_Network_weights_Boxing.h5"
Q_WEIGHTS = "./boxing/Q_Network_weights_Boxing.h5"

class DQVAgent:
	def __init__(self, action_size):
		self.render = True
		self.load_model = False
		self.state_size = (84, 84, 4)
		self.action_size = action_size

		self.q_model = self.build_q_model()
		self.value_model = self.build_v_model()

	def build_q_model(self):
		model = Sequential()
		model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=self.state_size))
		model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
		model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
		model.add(Flatten())
		model.add(Dense(512, activation='relu'))
		model.add(Dense(self.action_size))
		model.load_weights(Q_WEIGHTS)

		return model

	def build_v_model(self):
		model = Sequential()
		model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=self.state_size))
		model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
		model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
		model.add(Flatten())
		model.add(Dense(512, activation='relu'))
		model.add(Dense(1))
		model.load_weights(V_WEIGHTS)

		return model

	def get_expected_value(self, history):
		history = np.float32(history / 255.0)
		
		print("Expected Value for the current state {}:".format(self.value_model.predict(history)[0][0]))

	def get_action(self, history):
		history = np.float32(history / 255.0)

		q_value = self.q_model.predict(history)     

		print("Maximum Q value for best state-action pair {}:".format(max(q_value[0])))

		return np.argmax(q_value[0])

	def pre_processing(self, observe):
		processed_observe = np.uint8(resize(rgb2gray(observe), (84, 84), mode='constant') * 255)

		return processed_observe

if __name__ == "__main__":

	import tensorflow as tf
	
	env = gym.make(GAME)
	agent = DQVAgent(action_size=3)

	for e in range(EPISODES):

		done = False
		dead = False

		step, score, start_life = 0, 0, 5
		observe = env.reset()

		for _ in range(random.randint(1, 3)):
			observe, _, _, _ = env.step(1)

		state = agent.pre_processing(observe)
		history = np.stack((state, state, state, state), axis=2)
		history = np.reshape([history], (1, 84, 84, 4)) 

		while not done:
			
			if agent.render:
				env.render()

			step += 1

			agent.get_expected_value(history)
			action = agent.get_action(history)

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

			next_action = agent.get_action(next_history)	

			if start_life > info['ale.lives']:
				dead = True
				start_life = info['ale.lives']

			if dead:
				dead = False
			else:
				history = next_history
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="1"

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
EPISODES = 20000
V_WEIGHTS = "../icehockey/V_Network_weights_Hockey.h5"
Q_WEIGHTS = "../icehockey/Q_Network_weights_Hockey.h5"

class DeepDQVAgent:
	def __init__(self, action_size):

		self.render = True
		self.load_model = False
		self.state_size = (84, 84, 4)
		self.action_size = action_size

		self.learning_rate = 0.001
		self.discount_factor = 0.99

		self.no_op_steps = 30

		self.q_model = self.build_q_model()
		
		self.value_model = self.build_v_model()
		self.target_value_model = self.build_v_model()

		self.optimizer = self.optimizer()
	
	def update_v_target_model(self):
		self.target_value_model.set_weights(self.value_model.get_weights())

	def build_q_model(self):
		model = Sequential()
		model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=self.state_size))
		model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
		model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
		model.add(Flatten())
		model.add(Dense(512, activation='relu'))
		model.add(Dense(self.action_size))
		model.load_weights(Q_WEIGHTS)
		#model.compile(loss='mse', optimizer=RMSprop(lr=self.learning_rate, rho=0.95, epsilon=0.01))

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
		model.compile(loss='mse', optimizer=RMSprop(lr=self.learning_rate, rho=0.95, epsilon=0.01))

		return model

	def optimizer(self):

		a = K.placeholder(shape=(None,), dtype='int32')
		y = K.placeholder(shape=(None,), dtype='float32')

		py_x = self.q_model.output	#Q-values for the 3 different actions
	
		a_one_hot = K.one_hot(a, self.action_size)
		q_value = K.sum(py_x * a_one_hot, axis=1)
		
		error = K.abs(y - q_value)
	
		quadratic_part = K.clip(error, 0.0, 1.0)
		linear_part = error - quadratic_part

		loss = K.mean(0.5 * K.square(quadratic_part) + linear_part)

		optimizer = RMSprop(lr=0.00025, epsilon=0.01)
		updates = optimizer.get_updates(self.q_model.trainable_weights, [], loss)
		train = K.function([self.q_model.input, a, y], [loss], updates=updates)

		return train

	def save_model(self, name):
		self.model.save_weights(name)

	def get_action(self, history):

		history = np.float32(history / 255.0)

		q_value = self.q_model.predict(history)     

		return np.argmax(q_value[0])

	def pre_processing(self, observe):
		processed_observe = np.uint8(resize(rgb2gray(observe), (84, 84), mode='constant') * 255)

		return processed_observe

if __name__ == "__main__":

	import tensorflow as tf
	
	env = gym.make("IceHockeyDeterministic-v4")
	agent = DeepDQVAgent(action_size=3)

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
		history = np.reshape([history], (1, 84, 84, 4)) # We consider a state as 4 frames in a row
		
		# (1) State is initialized in history

		while not done:
			
			if agent.render:
				env.render()

			global_step += 1 
			step += 1

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

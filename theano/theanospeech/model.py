from __future__ import print_function

import os
import sys
import numpy as np
import timeit
import six.moves.cPickle as pickle
import gzip
import theano
import theano.tensor as T

from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d
__docformat__ = 'restructedtext en'

# start-snippet-1
class HiddenLayer(object):
	def __init__(self, rng, input, n_in, n_out, W=None, b=None,
				 activation=T.nnet.relu):
		self.input = input
		if W is None:
			W_values = np.asarray(
				#rng.uniform(
				#	low=-np.sqrt(6. / (n_in + n_out)),
				#	high=np.sqrt(6. / (n_in + n_out)),
				#	size=(n_in, n_out)
				#),
				rng.randn(n_in, n_out) * np.sqrt(6. / (n_in + n_out)),
				dtype=theano.config.floatX
			)
			if activation == theano.tensor.nnet.sigmoid:
				W_values *= 4
			W = theano.shared(value=W_values, name='W', borrow=True)
		if b is None:
			b_values = np.zeros((n_out,), dtype=theano.config.floatX)
			b = theano.shared(value=b_values, name='b', borrow=True)
		self.W = W
		self.b = b
		lin_output = T.dot(input, self.W) + self.b
		self.output = (
			lin_output if activation is None
			else activation(lin_output)
		)
		self.params = [self.W, self.b]

class LogisticRegression(object):
	def __init__(self, input, n_in, n_out):
		self.W = theano.shared(
			value=np.zeros(
				(n_in, n_out),
				dtype=theano.config.floatX
			),
			name='W',
			borrow=True
		)
		self.b = theano.shared(
			value=np.zeros(
				(n_out,),
				dtype=theano.config.floatX
			),
			name='b',
			borrow=True
		)
		self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
		self.y_pred = T.argmax(self.p_y_given_x, axis=1)
		self.params = [self.W, self.b]
		self.input = input

	def negative_log_likelihood(self, y):
		return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

class LeNetConvPoolLayer(object):
	def __init__(self, rng, input, filter_shape, image_shape, poolsize=(1, 2)):
		assert image_shape[1] == filter_shape[1]
		self.input = input
		fan_in = np.prod(filter_shape[1:])
		fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) //
				   np.prod(poolsize))
		W_bound = np.sqrt(6. / (fan_in + fan_out))
		self.W = theano.shared(
			np.asarray(
				#rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
				rng.randn(*filter_shape) * W_bound,
				dtype=theano.config.floatX
			),
			borrow=True
		)
		b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
		self.b = theano.shared(value=b_values, borrow=True)
		conv_out = conv2d(
			input=input,
			filters=self.W,
			filter_shape=filter_shape,
			input_shape=image_shape,
			border_mode=(filter_shape[2] / 2, filter_shape[3] / 2)
		)
		pooled_out = pool.pool_2d(
			input=conv_out,
			ds=poolsize,
			ignore_border=True
		)
		self.output = T.nnet.relu(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
		self.params = [self.W, self.b]
		self.input = input

class LeNetConvLayer(object):
	def __init__(self, rng, input, filter_shape, image_shape):
		assert image_shape[1] == filter_shape[1]
		self.input = input
		fan_in = np.prod(filter_shape[1:])
		fan_out = (filter_shape[0] * np.prod(filter_shape[2:]))
		W_bound = np.sqrt(6. / (fan_in + fan_out))
		self.W = theano.shared(
			np.asarray(
				#rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
				rng.randn(*filter_shape) * W_bound,
				dtype=theano.config.floatX
			),
			borrow=True
		)
		b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
		self.b = theano.shared(value=b_values, borrow=True)
		conv_out = conv2d(
			input=input,
			filters=self.W,
			filter_shape=filter_shape,
			input_shape=image_shape,
			border_mode=(filter_shape[2] / 2, filter_shape[3] / 2)
		)
		self.output = T.nnet.relu(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
		self.params = [self.W, self.b]
		self.input = input

class BatchNormalization(object) :
	def __init__(self, input_shape, mode=0 , momentum=0.9) :
		'''
		# params :
		input_shape :
			when mode is 0, we assume 2D input. (mini_batch_size, # features)
			when mode is 1, we assume 4D input. (mini_batch_size, # of channel, # row, # column)
		mode :
			0 : feature-wise mode (normal BN)
			1 : window-wise mode (CNN mode BN)
		momentum : momentum for exponential average
		'''
		self.input_shape = input_shape
		self.mode = mode
		self.momentum = momentum
		self.run_mode = 0 # run_mode : 0 means training, 1 means inference

		self.insize = input_shape[1]

		# random setting of gamma and beta, setting initial mean and std
		rng = np.random.RandomState(int(time.time()))
		self.gamma = theano.shared(np.asarray(rng.uniform(low=-1.0/math.sqrt(self.insize), high=1.0/math.sqrt(self.insize), size=(input_shape[1])), dtype=theano.config.floatX), name='gamma', borrow=True)
		self.beta = theano.shared(np.zeros((input_shape[1]), dtype=theano.config.floatX), name='beta', borrow=True)
		self.mean = theano.shared(np.zeros((input_shape[1]), dtype=theano.config.floatX), name='mean', borrow=True)
		self.var = theano.shared(np.ones((input_shape[1]), dtype=theano.config.floatX), name='var', borrow=True)

		# parameter save for update
		self.params = [self.gamma, self.beta]

	def set_runmode(self, run_mode) :
		self.run_mode = run_mode

	def get_result(self, input) :
		# returns BN result for given input.
		epsilon = 1e-06

		if self.mode==0 :
			if self.run_mode==0 :
				now_mean = T.mean(input, axis=0)
				now_var = T.var(input, axis=0)
				now_normalize = (input - now_mean) / T.sqrt(now_var+epsilon) # should be broadcastable..
				output = self.gamma * now_normalize + self.beta
				# mean, var update
				self.mean = self.momentum * self.mean + (1.0-self.momentum) * now_mean
				self.var = self.momentum * self.var + (1.0-self.momentum) * (self.input_shape[0]/(self.input_shape[0]-1)*now_var)
			else :
				output = self.gamma * (input - self.mean) / T.sqrt(self.var+epsilon) + self.beta

		else :
			# in CNN mode, gamma and beta exists for every single channel separately.
			# for each channel, calculate mean and std for (mini_batch_size * row * column) elements.
			# then, each channel has own scalar gamma/beta parameters.
			if self.run_mode==0 :
				now_mean = T.mean(input, axis=(0,2,3))
				now_var = T.var(input, axis=(0,2,3))
				# mean, var update
				self.mean = self.momentum * self.mean + (1.0-self.momentum) * now_mean
				self.var = self.momentum * self.var + (1.0-self.momentum) * (self.input_shape[0]/(self.input_shape[0]-1)*now_var)
			else :
				now_mean = self.mean
				now_var = self.var
			# change shape to fit input shape
			now_mean = self.change_shape(now_mean)
			now_var = self.change_shape(now_var)
			now_gamma = self.change_shape(self.gamma)
			now_beta = self.change_shape(self.beta)

			output = now_gamma * (input - now_mean) / T.sqrt(now_var+epsilon) + now_beta

		return output

	# changing shape for CNN mode
	def change_shape(self, vec) :
		return T.repeat(vec, self.input_shape[2]*self.input_shape[3]).reshape((self.input_shape[1],self.input_shape[2],self.input_shape[3]))

class Model():
	def sgd(self, params, grads, learning_rate):
		updates = [
			(param_i, param_i - learning_rate * grad_i)
			for param_i, grad_i in zip(params, grads)
		]
		return updates

	def rmsprop(self, l_rate, d_rate=0.9, epsilon=1e-6, parameters=None, grads=None):
		one = T.constant(1.0)

		def update_rule(param, cache, df):
			cache_val = d_rate * cache + (one-d_rate) * df**2
			x = l_rate * df / (T.sqrt(cache_val) + epsilon)
			updates = (param, param-x), (cache, cache_val)

			return updates

		caches = [theano.shared(name='c_{}'.format(param),
								value=param.get_value() * 0.,
								broadcastable=param.broadcastable)
				  for param in parameters]

		updates = []
		for p, c, g in zip(parameters, caches, grads):
			param_updates, cache_updates = update_rule(p, c, g)
			updates.append(param_updates)
			updates.append(cache_updates)

		return updates

	def adam(self, l_rate, beta1=0.9, beta2=0.999, epsilon=1e-6, parameters=None,
		 grads=None):
		one = T.constant(1.0)
		t = theano.shared(name='iteration', value=np.float32(1.0))

		def update_rule(param, moment, velocity, df):
			m_t = beta1 * moment + (one-beta1) * df
			v_t = beta2 * velocity + (one-beta2) * df**2
			m_hat = m_t/(one-beta1**(t))
			v_hat = v_t/(one-beta2**(t))
			x = (l_rate * m_hat / (T.sqrt(v_hat) + epsilon))
			updates = (param, param-x), (moment, m_t), (velocity, v_t)

			return updates

		moments = [theano.shared(name='m_{}'.format(param),
								 value=param.get_value() * 0.,
								 broadcastable=param.broadcastable)
				   for param in parameters]

		velocities = [theano.shared(name='v_{}'.format(param),
									value=param.get_value() * 0.,
									broadcastable=param.broadcastable)
					  for param in parameters]

		updates = []
		for p, m, v, g in zip(parameters, moments, velocities, grads):
			p_update, m_update, v_update = update_rule(p, m, v, g)
			updates.append(p_update)
			updates.append(m_update)
			updates.append(v_update)
		updates.append((t, t+1))

		return updates

	def __init__(self, **kwargs):
		self.kwargs = kwargs
		width = self.kwargs["width"]
		num_channels = self.kwargs["num_channels"]
		num_classes = self.kwargs["num_classes"]
		batch_size = self.kwargs["batch_size"]
		conv_settings = self.kwargs["conv_settings"]
		hidden_dims = self.kwargs["hidden_dims"]
		reg = self.kwargs["reg"]
		rng = np.random.RandomState(np.random.randint(10000))
		X = T.ftensor3('X')
		y = T.ivector('y')
		learning_rate = T.fscalar('learning_rate')
		layers = {}
		lastest_data = X.reshape((batch_size, num_channels, 1, width))
		lastest_channel = num_channels
		lastest_width = width
		
		for i in xrange(len(conv_settings)):
			F, C = conv_settings[i]
			layers['conv' + str(i)] = LeNetConvPoolLayer(
				rng,
				input = lastest_data,
				image_shape = (batch_size, lastest_channel, 1, lastest_width),
				filter_shape = (C, lastest_channel, 1, F)
			)
			lastest_data = layers['conv' + str(i)].output
			lastest_channel = C
			lastest_width /= 2
		
		lastest_data = lastest_data.flatten(2)

		for i in xrange(len(hidden_dims)):
			layers['fc' + str(i)] = HiddenLayer(
				rng,
				input = lastest_data,
				n_in = lastest_channel * lastest_width,
				n_out = hidden_dims[i],
				activation = T.tanh
			)
			lastest_data = layers['fc' + str(i)].output
			lastest_channel = 1
			lastest_width = hidden_dims[i]

		layers['reg'] = LogisticRegression(input = lastest_data, n_in = lastest_channel * lastest_width, n_out = num_classes)
		lastest_data = layers['reg'].y_pred

		params = []
		for (name, layer) in layers.items():
			params += layer.params
		
		cost = layers['reg'].negative_log_likelihood(y)
		for param in params:
			cost += 0.5 * reg * T.sum(param ** 2)

		grads = T.grad(cost, params)
		
		self.X = X
		self.y = y
		self.learning_rate = learning_rate
		self.batch_size = batch_size
		self.cost = cost
		self.params = params
		self.grads = grads
		self.lastest_data = lastest_data
		
	def compile(self):
		update = theano.function(
			inputs = [self.X, self.y, self.learning_rate],
			outputs = self.cost,
			updates = self.sgd(self.params, self.grads, self.learning_rate),
			#updates = self.adam(l_rate=self.learning_rate, parameters=self.params, grads=self.grads),
			#updates = self.rmsprop(l_rate=self.learning_rate, parameters=self.params, grads=self.grads),
			allow_input_downcast = True
		)
		
		predict = theano.function(
			inputs = [self.X],
			outputs = self.lastest_data,
			allow_input_downcast = True
		)
		
		return update, predict

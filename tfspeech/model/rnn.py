import numpy as np
import tensorflow as tf

def conv(X, F, C, weight_scale=0.01, reg_list=None, actfn=tf.nn.relu):
    _, width, iC = X.get_shape().as_list()
    
    W = tf.get_variable(shape=(F, iC, C), initializer=tf.random_normal_initializer(mean=0.1, stddev=weight_scale), dtype=tf.float32, name="W")
    b = tf.get_variable(shape=(C, ), initializer=tf.zeros_initializer, dtype=tf.float32, name="b")
    H = tf.nn.conv1d(X, W, stride=1, padding="SAME") + b
    if actfn is not None:
        H = actfn(H)
    if reg_list is not None:
        reg_list.append(tf.reduce_sum(W * W))
    return H

def flatten(X):
    _, width, channels = X.get_shape().as_list()
    H = tf.reshape(X, shape=(-1, width * channels))
    return H

def dense(X, dim, weight_scale=0.01, reg_list=None, actfn=tf.nn.relu):
    _, idim = X.get_shape().as_list()
    
    W = tf.get_variable(shape=(idim, dim), initializer=tf.random_normal_initializer(stddev=weight_scale), dtype=tf.float32, name="W")
    b = tf.get_variable(shape=(dim, ), initializer=tf.zeros_initializer, dtype=tf.float32, name="b")
    H = tf.matmul(X, W) + b
    if actfn is not None:
        H = actfn(H)
    if idim == dim:
        H = H + X
    if reg_list is not None:
        reg_list.append(tf.reduce_sum(W * W))
    return H

class Model:
	def __init__(self):
		self.kwargs = {
			"reg": 0.0,
			"optimizer": tf.train.GradientDescentOptimizer
		}

	def compile(self):
		self.graph = tf.Graph()

		width = self.kwargs["width"]
		num_channels = self.kwargs["num_channels"]
		num_neurons = self.kwargs["num_neurons"]
		reg = self.kwargs["reg"]
		optimizer = self.kwargs["optimizer"]
		
		with self.graph.as_default():
			X = tf.placeholder(dtype=tf.float32, shape=(None, width))
			y = tf.placeholder(dtype=tf.int32, shape=(None))
			
			Xfeed = tf.reshape(X, shape=(-1, width, 1))
			yfeed = tf.one_hot(y, depth=4)
			
			reg_list = []
			
			H = Xfeed
			with tf.variable_scope("Conv"):
				Hconv1 = conv(H, F=num_channels, C=3, reg_list=reg_list, actfn=tf.nn.relu)
				H = Hconv1
			with tf.variable_scope("RNN"):
				cell = tf.nn.rnn_cell.LSTMCell(num_neurons)
				Hrnn, _ = tf.nn.dynamic_rnn(cell, H, dtype=tf.float32)
				H = Hrnn
			Hflatten = flatten(H)
			H = Hflatten
			with tf.variable_scope('FullyConnect'):
				Y = dense(H, 4, actfn=None)
			
			if len(reg_list) == 0:
				reg_loss = 0.0
			elif len(reg_list) == 1:
				reg_loss = 0.5 * reg * reg_list[0]
			else:
				reg_loss = 0.5 * reg * tf.add_n(reg_list)
			softmax_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(Y, yfeed))
			loss = reg_loss + softmax_loss
			
			predict = tf.argmax(Y, axis=1)
			correct_predict = tf.equal(tf.to_int32(predict), y)
			accuracy = tf.reduce_mean(tf.to_float(correct_predict))
			
			lr = tf.placeholder(dtype=tf.float32, name="learning_rate")
			update = optimizer(lr).minimize(loss)

		self.node = {
			"X": X,
			"y": y,
			"update": update,
			"learning_rate": lr,
			"loss": loss,
			"accuracy": accuracy
		}

	def session(self):
		sess = tf.Session(graph=self.graph)
		with self.graph.as_default():
			sess.run(tf.global_variables_initializer())
		return sess

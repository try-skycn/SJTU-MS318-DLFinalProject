from tfspeech.model.__base__ import BaseModel
import numpy as np, tensorflow as tf

_optimizer_map = {
	"sgd": tf.train.GradientDescentOptimizer,
	"adam": tf.train.AdamOptimizer
}

def conv(X, width, channels, F, C, weight_scale, reg_list=None, actfn=tf.nn.relu):
	W = tf.get_variable(shape=(F, channels, C), initializer=tf.random_normal_initializer(stddev=weight_scale), dtype=tf.float32, name="W")
	b = tf.get_variable(shape=(C, ), initializer=tf.zeros_initializer, dtype=tf.float32, name="b")
	H = tf.nn.conv1d(X, W, stride=1, padding="VALID") + b
	if actfn is not None:
		H = actfn(H)
	if reg_list is not None:
		reg_list.append(tf.reduce_sum(W * W))
	return H, width - F + 1, C

def flatten(X, width, channels):
	dim = width * channels
	H = tf.reshape(X, shape=(-1, dim))
	return H, dim

def dense(X, input_dim, output_dim, weight_scale, reg_list=None, actfn=tf.nn.relu):
	W = tf.get_variable(shape=(input_dim, output_dim), initializer=tf.random_normal_initializer(stddev=weight_scale), dtype=tf.float32, name="W")
	b = tf.get_variable(shape=(output_dim, ), initializer=tf.zeros_initializer, dtype=tf.float32, name="b")
	H = tf.matmul(X, W) + b
	if actfn is not None:
		H = actfn(H)
	if reg_list is not None:
		reg_list.append(tf.reduce_sum(W * W))
	return H, output_dim

class Model(BaseModel):
	def setdefault(self):
		self.kwargs.setdefault("weight_scale", 0.001)
		self.kwargs.setdefault("reg", 0.001)
		self.kwargs.setdefault("optimizer", "sgd")

	def setname(self):
		pass

	def build(self):
		weight_scale = self.kwargs["weight_scale"]
		reg = self.kwargs["reg"]
		optimizer = self.kwargs["optimizer"]
		width = self.kwargs["width"]
		num_channels = self.kwargs["num_channels"]
		num_classes = self.kwargs["num_classes"]
		conv_settings = self.kwargs["conv_settings"]
		hidden_dims = self.kwargs["hidden_dims"]

		reg_list = []

		with tf.name_scope("Data"):
			X = tf.placeholder(shape=(None, width, num_channels), dtype=tf.float32, name="X")
			y = tf.placeholder(shape=(None, num_classes), dtype=tf.float32, name="y")

		with tf.variable_scope("Conv0"):
			F, C = conv_settings[0]
			H, hidden_width, hidden_channels = conv(X, width, num_channels, F, C, weight_scale, reg_list=reg_list)

		for i, (F, C) in enumerate(conv_settings[1:], start=1):
			with tf.variable_scope("Conv{}".format(i)):
				H, hidden_width, hidden_channels = conv(H, hidden_width, hidden_channels, F, C, weight_scale, reg_list=reg_list)

		H, hidden_dim = flatten(H, hidden_width, hidden_channels)

		for i, D in enumerate(hidden_dims):
			with tf.variable_scope("Dense{}".format(i)):
				H, hidden_dim = dense(H, hidden_dim, D, weight_scale, reg_list=reg_list)

		Y, hidden_dim = dense(H, hidden_dim, num_classes, weight_scale, reg_list=reg_list, actfn=None)

		reg_loss = (0.5 * reg) * tf.add_n(reg_list)
		softmax_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(y, Y))
		loss = softmax_loss + reg_loss

		learning_rate = tf.placeholder(dtype=tf.float32, name="learning_rate")
		update = _optimizer_map[optimizer](learning_rate).minimize(loss)

		self.X = X
		self.y = y
		self.Y = Y
		self.learning_rate = learning_rate
		self.loss = loss
		self.update = update

import numpy as np
import tensorflow as tf

def dense(X, dim, reg_list=None, actfn=tf.nn.relu):
	_, idim = X.get_shape().as_list()
	
	weight_scale = np.sqrt(6.0/(idim+dim))
	W = tf.get_variable(shape=(idim, dim), initializer=tf.random_uniform_initializer(minval=-weight_scale, maxval=weight_scale), dtype=tf.float32, name="W")
	b = tf.get_variable(shape=(dim, ), initializer=tf.zeros_initializer, dtype=tf.float32, name="b")
	H = tf.matmul(X, W) + b
	if actfn is not None:
		H = actfn(H)
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
		reg = self.kwargs["reg"]
		optimizer = self.kwargs["optimizer"]
		
		with self.graph.as_default():
			X = tf.placeholder(dtype=tf.float32, shape=(None, width))
			y = tf.placeholder(dtype=tf.int32, shape=(None))
			
			yfeed = tf.one_hot(y, depth=4)
			
			reg_list = []
			
			with tf.variable_scope('FullyConnect'):
				Y = dense(X, 4, reg_list=reg_list, actfn=None)
			
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

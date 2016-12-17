from tfspeech.model.__base__ import BaseModel, BaseInstance
import numpy as np, tensorflow as tf

_optimizer_map = {
	"sgd": tf.train.GradientDescentOptimizer,
	"adam": tf.train.AdamOptimizer
}

def interval(X, a, b, scale=0.2, name="Interval"):
	with tf.variable_scope(name):
		scale = 6.0 / ((b - a) * scale)
		a *= scale
		b *= scale

		scale = tf.get_variable("scale", dtype=tf.float32, initializer=scale)
		a = tf.get_variable("a", dtype=tf.float32, initializer=a)
		b = tf.get_variable("b", dtype=tf.float32, initializer=b)

		X = X * scale
		bottom = tf.nn.relu6(X - a)
		top = tf.nn.relu6(- X + b)

		return bottom * top

def derivative(X, filter_size, name="Derivative"):
	with tf.variable_scope(name):
		half_filter = filter_size // 2
		W = np.array([-1., ] * half_filter + [0., ] + [1., ] * half_filter, dtype=np.float32).reshape(filter_size, 1, 1) / filter_size
		W = tf.get_variable("W", dtype=tf.float32, initializer=W)

		return tf.nn.conv1d(X, W, stride=1, padding="SAME")

class Model(BaseModel):
	def setdefault(self):
		self.kwargs.setdefault("weight_scale", 0.001)
		self.kwargs.setdefault("optimizer", "sgd")

	def setname(self):
		pass

	def compile(self):
		return ModelInstance(self)

	def build(self):
		weight_scale = self.kwargs["weight_scale"]
		optimizer = self.kwargs["optimizer"]
		width = self.kwargs["width"]

		with tf.name_scope("Data"):
			X = tf.placeholder(shape=(None, width, 1), dtype=tf.float32, name="X")
			y = tf.placeholder(shape=(None, 4), dtype=tf.float32, name="y")

		Xp = derivative(X, 3, name="FirstOrder")
		Xpp = derivative(Xp, 3, name="SecondOrder")
		Xppp = derivative(Xpp, 3, name="ThirdOrder")

		valid = interval(X, 50, 300, name="Valid")
		good = interval(Xppp, -5, 5, name="Good")

		flat = interval(Xp, -3, 3, name="Flat") * valid * good
		goup = interval(Xp, 3, 10, name="GoUp") * valid * good
		godown = interval(Xp, -10, -3, name="GoDown") * valid * good
		valley = interval(Xpp, 5, 20, name="Valley") * valid * good

		feature = tf.pack([flat, goup, valley, godown], axis=1)
		feature = tf.reduce_mean(feature, axis=[2, 3])

		with tf.variable_scope("Final"):
			W = tf.get_variable(dtype=tf.float32, initializer=np.float32(np.identity(4) + np.random.randn(4, 4) * weight_scale), name="W")
			Y = tf.matmul(feature, W)

		loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(Y, y))

		learning_rate = tf.placeholder(dtype=tf.float32, name="learning_rate")
		update = _optimizer_map[optimizer](learning_rate).minimize(loss)

		self.X = X
		self.y = y
		self.Y = Y
		self.learning_rate = learning_rate
		self.loss = loss
		self.update = update

class ModelInstance(BaseInstance):
	def update(self, X, y, learning_rate):
		return super(ModelInstance, self).update(X[:, :, 0:1], y, learning_rate)

	def predict(self, X):
		return super(ModelInstance, self).predict(X[:, :, 0:1])

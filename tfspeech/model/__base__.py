import numpy as np, tensorflow as tf

class BaseModel:
	def __init__(self, **kwargs):
		self.kwargs = kwargs
		self.setdefault()

		self.name = None
		self.setname()

		self.graph = tf.Graph()
		with self.graph.as_default():
			self.build()

	def setdefault(self):
		pass

	def setname(self):
		pass

	def build(self):
		pass

	def compile(self):
		return BaseInstance(self)

class BaseInstance:
	def __init__(self, model):
		self.model = model
		self.sess = tf.Session(graph=self.model.graph, config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
		with self.model.graph.as_default():
			self.sess.run(tf.global_variables_initializer())

	def update(self, X, y, learning_rate):
		_, loss = self.sess.run([self.model.update, self.model.loss], feed_dict={self.model.learning_rate: learning_rate, self.model.X: X, self.model.y: y})
		return loss

	def predict(self, X):
		Y = self.sess.run(self.model.Y, feed_dict={self.model.X: X})
		return Y
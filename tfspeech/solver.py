import numpy as np

class Solver:
	def __init__(self, data, model):
		self.data = data
		self.model = model
		self.kwargs = {}

	def train(self):
		batch_size = self.kwargs["batch_size"]
		epoch_size = self.kwargs["epoch_size"]
		epochs = self.kwargs["epochs"]

		lr_it = self.kwargs["learning_rate"]
		lr_decay = self.kwargs["learning_rate_decay"]
		print_every = self.kwargs["print_every"]

		self.sess = self.model.session()
		node_X = self.model.node["X"]
		node_y = self.model.node["y"]
		node_update = self.model.node["update"]
		node_loss = self.model.node["loss"]
		node_accuracy = self.model.node["accuracy"]
		node_learning_rate = self.model.node["learning_rate"]

		try:
			for e in range(epochs):
				loss_list = []
				for it in range(epoch_size):
					X, y = self.data.sample("train", batch_size)
					_, loss = self.sess.run([node_update, node_loss], feed_dict={node_X: X, node_y: y, node_learning_rate: lr_it})
					loss_list.append(loss)
				if (e + 1) % print_every == 0:
					mean_loss = np.mean(loss_list)
					loss_list.clear()
					X, y = self.data.get("train")
					train_accuracy = self.sess.run(node_accuracy, feed_dict={node_X: X, node_y: y})
					X, y = self.data.get("test")
					test_accuracy = self.sess.run(node_accuracy, feed_dict={node_X: X, node_y: y})
					X, y = self.data.get("test_new")
					test_new_accuracy = self.sess.run(node_accuracy, feed_dict={node_X: X, node_y: y})
					print(
						"Epoch: {}, lr {:.4f}, loss {:.4f}, train acc {:.2f}%, test acc {:.2f}%, test_new acc {:.2f}%".format(
							e+1, lr_it, mean_loss, train_accuracy, test_accuracy, test_new_accuracy
						)
					)
				lr_it = lr_it * lr_decay
		except KeyboardInterrupt:
			print("Stop")

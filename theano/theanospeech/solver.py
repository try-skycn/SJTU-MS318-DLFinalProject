import numpy as np

def _apply_color(color_idx, s):
	return "\033[{color_idx};1m{s}\033[m".format(color_idx=color_idx, s=s)

def _apply_format(color_idx, fieldname, *args):
	return "{fieldname} {args}".format(
				fieldname=_apply_color(color_idx, "[{}] ==>".format(fieldname)),
				args=' '.join(
							["{}: {}".format(x, y) for x, y in args]
						)
			)

_color_map = {
	"step": 32,
	"train": 33,
	"test": 33,
	"test_new": 31
}

class Solver:
	def __init__(self, data, update, predict, **kwargs):
		self.data = data
		self.update = update
		self.predict = predict
		self._batch_size = kwargs.pop("batch_size")
		self._learning_rate = kwargs.pop("learning_rate")
		self._num_epochs = kwargs.pop("num_epochs")
		self._verbose = kwargs.pop("verbose", False)
		self.history = {}

	def _push_history(self, fieldname, step, *args):
		self.history.setdefault(fieldname, {})
		fieldhistory = self.history[fieldname]
		for metricname, metricvalue in args:
			fieldhistory.setdefault(metricname, [])
			fieldhistory[metricname].append((step, metricvalue))
		self._print_history(fieldname, *args)

	def _print_step(self, step):
		if self._verbose >= 0:
			print(_apply_color(_color_map["step"], "step") + " {}".format(step))

	def _print_history(self, fieldname, *args):
		if self._verbose:
			print(_apply_format(_color_map[fieldname], fieldname, *args))

	def _update(self, X, y):
		return float(self.update(X, y, learning_rate=self._learning_rate))

	def _accuracy(self, X, y):
		result = 0
		for i in xrange((X.shape[0] + self._batch_size - 1) / self._batch_size):
			if i == (X.shape[0] + self._batch_size - 1) / self._batch_size - 1:
				result += float((self.predict(X[X.shape[0] - self._batch_size : X.shape[0]])[self._batch_size - (X.shape[0] - self._batch_size * i) : self._batch_size] == y[self._batch_size * i : X.shape[0]]).mean()) * (X.shape[0] - self._batch_size * i) / X.shape[0]
			else:
				result += float((self.predict(X[self._batch_size * i : self._batch_size * (i + 1)]) == y[self._batch_size * i : self._batch_size * (i + 1)]).mean()) * self._batch_size / X.shape[0]
		return result

	def _test(self, X, y):
		for i in xrange(X.shape[0]):
			list = []
			for j in xrange(self._batch_size):
				if j == 0:
					list.append(X[i])
				else:
					list.append(np.zeros_like(X[i]))
			pred = self.predict(np.array(list))[0]
			if pred != y[i]:
				print 'error at', X[i], y[i], pred

	def train(self):
		step = 0
		self.best_test = 0
		self.best_test_new = 0
		for epochidx in range(1, self._num_epochs + 1):
			for X, y in self.data.get("train"):
				step += 1
				if step % 100 == 0:
					self._print_step(step)

				loss = self._update(X, y)
				accuracy = self._accuracy(X, y)
				#if step % 100 == 0:
					#self._push_history("train", step, ("loss", loss), ("accuracy", accuracy))

			accuracy = self._accuracy(*self.data.get("test"))
			if accuracy > self.best_test:
				self.best_test = accuracy
				self.best_test_t = epochidx
			self._push_history("test", step, ("accuracy", accuracy))

			accuracy = self._accuracy(*self.data.get("test_new"))
			if accuracy > self.best_test_new:
				self.best_test_new = accuracy
				self.best_test_new_t = epochidx
			self._push_history("test_new", step, ("accuracy", accuracy))

			#if epochidx % 100 == 0:
			#	self._test(*self.data.get("test_new"))

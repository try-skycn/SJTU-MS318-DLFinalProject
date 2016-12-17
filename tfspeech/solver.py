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
	def __init__(self, data, modelinst, **kwargs):
		self.data = data
		self.modelinst = modelinst
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
		return float(self.modelinst.update(X, y, learning_rate=self._learning_rate))

	def _accuracy(self, X, y):
		return float((self.modelinst.predict(X).argmax(1) == y.argmax(1)).mean())

	def train(self):
		step = 0
		for epochidx in range(1, self._num_epochs + 1):
			for X, y in self.data.get("train"):
				step += 1
				self._print_step(step)

				loss = self._update(X, y)
				accuracy = self._accuracy(X, y)
				self._push_history("train", step, ("loss", loss), ("accuracy", accuracy))

			accuracy = self._accuracy(*self.data.get("test"))
			self._push_history("test", step, ("accuracy", accuracy))

		accuracy = self._accuracy(*self.data.get("test_new"))
		self._push_history("test_new", step, ("accuracy", accuracy))

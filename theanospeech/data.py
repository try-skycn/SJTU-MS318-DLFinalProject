import os, numpy as np
import json

_pkgname_list = ["train", "test", "test_new"]

def _readpkg(pkgname, data):
	X = np.array([data[pkgname][i]['input'] for i in xrange(len(data[pkgname]))], dtype=np.float)
	print X.shape
	X = X.reshape((X.shape[0], 1, X.shape[1]))
	y = np.array([data[pkgname][i]['label'] - 1 for i in xrange(len(data[pkgname]))], dtype=np.float)
	return X, y

class Data:
	def __init__(self, filedir, **kwargs):
		self.width = kwargs.pop("width")
		self.batch_size = kwargs.pop("batch_size")
		self.epoch_size = kwargs.pop("epoch_size")

		self._data = {}
		datafile = file(filedir)
		data = json.load(datafile)
		for pkgname in _pkgname_list:
			self._data[pkgname] = _readpkg(pkgname, data)

	def _get_train(self):
		X, y = self._data["train"]
		size = X.shape[0]
		for _ in range(self.epoch_size):
			mask = np.random.choice(size, self.batch_size)
			yield X[mask] + np.random.randn(*X[mask].shape) * 0.00, y[mask]

	def get(self, pkgname):
		if pkgname == "train":
			return self._get_train()
		else:
			return self._data[pkgname]

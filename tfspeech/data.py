import os, numpy as np

_pkgname_list = ["train", "test", "test_new"]
_label_word = ["one", "two", "three", "four"]
def _prepare_index(dirname, *args):
	result = {}
	for pkgname in args:
		pkgdir = os.path.join(dirname, pkgname)
		pkglist = []
		for i, word in enumerate(_label_word):
			labeldir = os.path.join(pkgdir, word)
			for filename in os.listdir(labeldir):
				if filename[-3:] == ".f0":
					pkglist.append((
							os.path.join(labeldir, filename),
							os.path.join(labeldir, filename[:-3] + ".engy"),
							i
						))
		result[pkgname] = pkglist
	return result

def _readfile(filename, size):
	result = np.zeros(size, dtype=np.float)
	with open(filename, "r") as f:
		for i, line in enumerate(f):
			result[i] = float(line.strip())
	return result

_identity = np.identity(len(_label_word), dtype=np.float)
def _readentry(f0, engy, size, label):
	X = np.zeros((size, 2), dtype=np.float)
	X[:, 0] = _readfile(f0, size)
	X[:, 1] = _readfile(engy, size)
	return X, _identity[label]

def _readpkg(pkglist, size):
	X = np.zeros((len(pkglist), size, 2), dtype=np.float)
	y = np.zeros((len(pkglist), len(_label_word)), dtype=np.float)
	for i, (f0, engy, label) in enumerate(pkglist):
		X[i], y[i] = _readentry(f0, engy, size, label)
	return X, y

class Data:
	def __init__(self, dirname, **kwargs):
		self.width = kwargs.pop("width")
		self.batch_size = kwargs.pop("batch_size")
		self.epoch_size = kwargs.pop("epoch_size")

		self._data = {}
		pkgdir_map = _prepare_index(dirname, *_pkgname_list)
		for pkgname, pkglist in pkgdir_map.items():
			self._data[pkgname] = _readpkg(pkglist, self.width)

	def _get_train(self):
		X, y = self._data["train"]
		size = X.shape[0]
		for _ in range(self.epoch_size):
			mask = np.random.choice(size, self.batch_size)
			yield X[mask], y[mask]

	def get(self, pkgname):
		if pkgname == "train":
			return self._get_train()
		else:
			return self._data[pkgname]

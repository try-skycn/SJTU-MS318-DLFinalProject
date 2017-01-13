class Data:
	def __init__(self, filename):
		with open(filename, 'r') as fi:
			self.data = json.load(fi)
		for k in data:
			f0, label = [], []
			for x in self.data[k]:
				f0.append(x['f0'])
				label.append(x['label'])
			self.data[k] = (np.array(f0), np.array(label))

	def get(self, key):
		return self.data[key]

	def sample(self, key, batch_size):
		mask = np.random.choice(len(self.data[key][0]), batch_size)
		return self.data[key][0][mask], self.data[key][1][mask]

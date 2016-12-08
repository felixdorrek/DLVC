import numpy as np
from collections import Counter 

class ClassificationDataset:

	def size(self):
		return len(self.labels)

	def nclasses(self):
		return len({l for l in self.labels})

	def size_of_class(self, cid):
		cnt = Counter()
		for l in self.labels:
			cnt[l]+=1
		return cnt[cid]

	def classname(self, cid):
		return self.class_names[cid]

	def sample(self, sid):
		sample = self.samples[sid, :]
		label = self.labels[sid]
		return {"class_id" : label, "sample" : sample}

	def sample_shape(self):
		return self.samples[0, :].shape

	def get_samples(self):
		return self.samples

	def get_labels(self):
		return np.array(self.labels)

	def set_samples(self, samples):
		self.samples = samples

	def set_labels(self, labels):
		self.labels = labels

	def get_classnames(self):
		return self.class_names



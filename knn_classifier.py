from collections import Counter
import numpy as np

class KnnClassifier:

	def __init__(self, k, cmp):
		self.k = k
		self.cmp = cmp 


	def train(self, dataset):
		self.fvecs = [v.astype(float) for v in dataset.data]
		self.norms = [np.dot(v,v) for v in self.fvecs]
		self.labels = dataset.labels


	def predict(self, fvec):
		fvecs = self.fvecs
		norm = np.dot(fvec, fvec)

		if self.cmp == "l2":
			dists = [self.norms[i] - 2*np.dot(fvecs[i], fvec) + norm for i in range(len(fvecs))]
		else:
			dists = [np.sum(np.abs(fvec - fvecs[i])) for i in range(len(fvecs))]

		inds = np.array(dists).argsort()
		cnt = Counter()
		for j in range(self.k):
			label = self.labels[inds[j]]
			cnt[label] += 1
		return cnt.most_common(1)[0][0]

	def score(self, dataset):
		n_correct_predictions = 0
		n_samples = dataset.size()
		for i in range(n_samples):
			if i % 250 == 0: print "  Scoring samples {0} - {1} ...".format(i+1, min(i+250, n_samples))
			sample = dataset.sample(i)
			pred_cid = self.predict(sample["sample"])
			cid = sample["class_id"]
			if pred_cid == cid: 
				n_correct_predictions +=1
		return float(n_correct_predictions)/float(n_samples)
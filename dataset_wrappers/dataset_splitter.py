from collections import Counter

class Splitter:

	def __init__(self, data, labels):
		self.data = data
		self.labels = labels

	def nj_per_class(self, numbers, order):
		# Returns an array with the first/last number[j] samples of classes j,
		# 0<=j<=n_classes, in an order preserving way.
		if order == "first":
			indices = self.get_first_nj_indices_per_class(numbers)
			return self.data[indices, :], [self.labels[i] for i in indices]
		elif order == "last":
			indices = self.get_last_nj_indices_per_class(numbers)
			return self.data[indices, :], [self.labels[i] for i in indices]

	def get_first_nj_indices_per_class(self, numbers):
		cnt = Counter()
		indices = tuple()
		for index, j in enumerate(self.labels):
			cnt[j] += 1
			if (cnt[j] <= numbers[j]): indices += (index,)
		return indices

	def get_last_nj_indices_per_class(self, numbers):
		cnt = Counter()
		indices = tuple()
		for index, j in enumerate(reversed(self.labels)):
			cnt[j] +=1
			if (cnt[j] <= numbers[j]): indices = (len(self.labels) -1 -index, ) + indices
		return indices
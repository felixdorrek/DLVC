import numpy as np
import h5py
from classification_dataset import ClassificationDataset

class FeatureVectorDataset(ClassificationDataset):
	pass


class ImageVectorizer(FeatureVectorDataset):

	def __init__(self,dataset):

		self.dataset = dataset
		try:
			samples = dataset.get_samples()
			self.samples = np.array([self.vectorize(samples[j,:,:,:]) for j in range(dataset.size())] )
			self.labels = dataset.get_labels()
			self.class_names = dataset.get_classnames()
		except:
			self._load()

	def vectorize(self, sample):
		return sample.reshape(sample.size)

	def devectorize(self, fvec):
		return fvec.reshape(self.dataset.sample_shape())

	def _load(self):
		self.dataset.load()
		samples = self.dataset.get_samples()
		self.samples = np.array([self.vectorize(samples[j,:,:,:]) for j in range(self.dataset.size())])
		self.labels = self.dataset.get_labels()
		self.class_names = self.dataset.get_classnames()


class HDF5FeatureVectorDataset(FeatureVectorDataset):

	def __init__(self, fpath, class_names):
		f = h5py.File(fpath, "r")
		self.data = np.array(f.get('features'))
		self.labels = np.array(f.get('labels'))
		self.class_names = class_names


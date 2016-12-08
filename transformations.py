import numpy as np


class SampleTransformation:

	def apply(self, sample):
		pass

	def apply_to_dataset(self, dataset):

		samples = dataset.get_samples()
		transformation = lambda s : self.apply(s)
		dataset.set_samples( np.array( [ transformation(samples[j,:,:,:]) for j in range(dataset.size()) ] ))
		#dataset.set_samples(np.apply_over_axes(transformation, samples, range(0,len(dataset.sample_shape()))))


class IdentityTransformation(SampleTransformation):

	def apply(self, sample):
		return sample


class FloatCastTransformation(SampleTransformation):

	def apply(self, sample):
		return sample.astype(np.float32)
	

class SubtractionTransformation(SampleTransformation):

	@staticmethod
	def from_dataset_mean(dataset, tform=None):
		value =  dataset.get_samples().mean()
		return SubtractionTransformation(value)

	def __init__(self, value):
		self.val = value
		self.transformation = np.vectorize(lambda x: x - value)

	def apply(self, sample):
		return self.transformation(sample)

	def value(self):
		return self.val


class DivisionTransformation(SampleTransformation):

	@staticmethod
	def from_dataset_stddev(dataset, tform=None):
		value =  dataset.get_samples().std()
		return DivisionTransformation(value)

	def __init__(self, value):
		assert value != 0
		self.val = value
		self.transformation = np.vectorize(lambda x: x/value)

	def apply(self, sample):
		return self.transformation(sample)

	def value(self):
		return self.val


class TransformationSequence(SampleTransformation):

	def __init__(self, transformations=[]):
		self.transformations = transformations

	def add_transformation(self, transformation):
		self.transformations.append(transformation)

	def get_transformation(self, tid):
		return self.transformations[tid]

	def apply(self, sample):
		retval = sample 
		for transformation in self.transformations:
			retval = transformation.apply(retval)
		return retval


class PerChannelSubtractionImageTransformation(SampleTransformation):

	@staticmethod
	def from_dataset_mean(dataset, tform=None):
		values = [dataset.get_samples_by_channel(j).mean() for j in range(3)]
		return PerChannelSubtractionImageTransformation(values=values)

	def __init__(self, values=[0,0,0]):
		self.vals = values
		self.transformation =  self._build_transformation(values)


	def _build_transformation(self,values):
		#traffos = [ np.vectorize( lambda x, j=0: x - values[j] ) for j in range(3)]
		t0 = np.vectorize( lambda x: x -values[0])
		t1 = np.vectorize( lambda x: x-values[1])
		t2 = np.vectorize( lambda x: x-values[2])
		traffos = [t0, t1, t2]
		return lambda sample: np.stack([ traffos[j](sample[:,:,j]) for j in range(3)], axis=2)

	def apply(self, sample):
		return self.transformation(sample)

	def values(self):
		return self.vals


class PerChannelDivisionImageTransformation(SampleTransformation):

	@staticmethod
	def from_dataset_stddev(dataset, tform=None):
		values = [dataset.get_samples_by_channel(j).std() for j in range(3)]
		return PerChannelDivisionImageTransformation(values=values)

	def __init__(self, values=[1,1,1]):
		self.vals = values
		self.transformation = self._build_transformation(values)


	def _build_transformation(self,values):
		#traffos = [np.vectorize(lambda x, j=0: x/values[j]) for j in range(3)]
		t0 = np.vectorize( lambda x: x/values[0])
		t1 = np.vectorize( lambda x: x/values[1])
		t2 = np.vectorize( lambda x: x/values[2])
		traffos = [t0, t1, t2]
		return lambda sample: np.stack([ traffos[j](sample[:,:,j]) for j in range(3)], axis=2)

	def apply(self, sample):
		return self.transformation(sample)

	def values(self):
		return self.vals




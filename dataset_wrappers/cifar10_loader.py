import cPickle
import numpy as np
from dataset_splitter import Splitter 
import time 

class Cifar10Loader:

	# call action returns a dictionary with the classnames and the labels and data 
	# of the specified split
	# split=test returns the first (100*prop)% samples of the test data
	# split=train return the first (prop*(1-val_prop)*100)% of the train/val_data
	# split=val return the first (100*prop)% of the last (val_prop*100)% of 
	#the train/val_data

	def __init__(self, fdir, split, prop=1, val_prop=0.2):
		self.fdir = fdir
		self.split = split
		self.prop = prop
		self.train_size = 50000 * prop * (1.0-val_prop)
		self.val_size = 50000 * prop * val_prop
		self.test_size = 10000 * prop

	def call(self):
		class_names = { 
			j : label_name for j, label_name in enumerate(self._load_names("batches.meta"))
			}
		if self.split == 'train':
			complete_data, labels = self._load_complete()
			data, labels = Splitter(
				complete_data, labels
				).nj_per_class([4000]*10, "first")
			if self.prop <1: data, labels = Splitter(data, labels).nj_per_class([self.train_size/10]*10, "first")
		elif self.split == 'val':
			complete_data, labels = self._load_complete()
			data, labels = Splitter(
				complete_data, labels
				).nj_per_class([1000]*10, "last")
			if self.prop <1: data, labels = Splitter(data, labels).nj_per_class([self.val_size/10]*10, "first")
		elif	self.split == 'test':
			data, labels = self._load_batch("test_batch")
			data = data.reshape(data.shape[0],32,32,3, order="F").transpose(0,2,1,3)
			if self.prop <1: data, labels = Splitter(data, labels).nj_per_class([self.test_size/10]*10, "first")

		
		return {"data":data, "labels":labels, "class_names":class_names}

	def _load_batch(self,file_name):
		connection = open(self.fdir +"/"+file_name, 'rb')
		dictionary = cPickle.load(connection)
		connection.close()
		return dictionary["data"], dictionary["labels"]		

	def _load_names(self, file_name):
		connection = open(self.fdir +"/"+file_name, 'rb')
		dictionary = cPickle.load(connection)
		connection.close()
		return dictionary["label_names"]	

	def _load_complete(self):
		data = tuple()
		labels = []
		for i in range(5):
			batch = self._load_batch("data_batch_{0}".format(i+1))
			data+= batch[0],
			labels+= batch[1]
		complete_data = np.concatenate(data)
		complete_data = complete_data.reshape(complete_data.shape[0],32,32,3, order="F").transpose(0,2,1,3)
		return complete_data, labels

	def _helper(self,vec):
		return np.concatenate( [vec[[j,j+1024,j+2048]] for j in range(1024)] )
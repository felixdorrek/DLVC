import numpy as np
from collections import Counter
from cifar10_loader import Cifar10Loader
from classification_dataset import ClassificationDataset

class ImageDataset(ClassificationDataset):
	
	def get_samples_by_channel(self, channel):
		return self.samples[:,:,:,channel]


class Cifar10Dataset(ImageDataset):

	def __init__(self, fdir, split):
		self.fdir = fdir
		self.split = split

	def load(self):
		dictionary = Cifar10Loader(self.fdir, self.split).call()
		self.labels = dictionary["labels"]
		self.samples = dictionary["data"]
		self.class_names = dictionary["class_names"]


class TinyCifar10Dataset(ImageDataset):

	def __init__(self, fdir, split):
		self.fdir = fdir
		self.split = split

	def load(self):
		dictionary = Cifar10Loader(self.fdir, self.split, prop=0.1).call()
		self.labels = dictionary["labels"]
		self.samples = dictionary["data"]
		self.class_names = dictionary["class_names"]





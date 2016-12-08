from dataset_wrappers.image_datasets import TinyCifar10Dataset
from dataset_wrappers.feature_vector_datasets import ImageVectorizer
import numpy as np 
from mini_batch_generator import MiniBatchGenerator

image_dataset = TinyCifar10Dataset("datasets/cifar-10-batches-py", "train")
image_dataset.load()
feature_vector_dataset = ImageVectorizer(TinyCifar10Dataset("datasets/cifar-10-batches-py", "train"))

def test(dataset):
	generator = MiniBatchGenerator(dataset, 60)
	print "Dataset has {0} samples".format(dataset.size())
	print "Batch generator has {0} minibatches, minibatch size: {1}".format(generator.nbatches(), generator.batchsize())

 	batch_0 = generator.batch(0)
	print "Minibatch #0 has {0} samples".format(len(batch_0["labels"]))
	print " Data shape: {0}".format(batch_0["samples"].shape)
	print " First 10 sample IDs: {0}".format(batch_0["ids"][0:10])

	batch_66 = generator.batch(66)
	print "Minibatch #66 has {0} samples".format(len(batch_66["labels"]))
	print " First 10 sample IDs: {0}".format(batch_66["ids"][0:10])

	print "Shuffling samples"

	generator.shuffle()

	batch_0 = generator.batch(0)
	print "Minibatch #0 has {0} samples".format(len(batch_0["labels"]))
	print " First 10 sample IDs: {0}".format(batch_0["ids"][0:10])

	batch_66 = generator.batch(66)
	print "Minibatch #66 has {0} samples".format(len(batch_66["labels"]))
	print " First 10 sample IDs: {0}".format(batch_66["ids"][0:10])


print "=== Testing with TinyCifar10Dataset ==="
test(image_dataset)

print "=== Testing with ImageVectorizer ==="
test(feature_vector_dataset)
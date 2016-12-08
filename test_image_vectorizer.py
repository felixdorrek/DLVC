from dataset_wrappers import feature_vector_datasets, image_datasets

dataset = image_datasets.TinyCifar10Dataset("datasets/cifar-10-batches-py", "train")
dataset.load()
vectorizer = feature_vector_datasets.ImageVectorizer(dataset)

print "{0} samples".format(vectorizer.size())
print "{0} classes, ".format(vectorizer.nclasses()) + "name of class #1: {0}".format(vectorizer.classname(1))
sample = vectorizer.sample(499)["sample"]
sample_cid = vectorizer.sample(499)["class_id"]
print "Sample #499: {0}, shape: {1}".format(vectorizer.classname(sample_cid), sample.shape)
image_sample = vectorizer.devectorize(sample)
print "Shape after devectorization: {0}".format(image_sample.shape)
from dataset_wrappers import image_datasets
from collections import Counter
from PIL import Image

for split in ["train", "val", "test"]:

	dataset = image_datasets.TinyCifar10Dataset("datasets/cifar-10-batches-py", split)

	dataset.load()


	print "[{0}] ".format(split)+str(dataset.nclasses())+" classes, name of class #1: " + str(dataset.classname(1))

	print " "

	print "[{0}] ".format(split) + str(dataset.size())+ " samples"
	for j in range(10):
		print " Class #{0}: ".format(j) +str(dataset.size_of_class(j)) + " samples"

	print " "

	sample = dataset.sample(499)
	im = Image.fromarray(sample["sample"])
	dest = "{0}_tiny_sample_500.png".format(split)
	im.save(dest)

	print"[{0}] Sample #499:".format(split) + str(dataset.classname(sample["class_id"])) 

	print " "
	print "---"*30
	print " "
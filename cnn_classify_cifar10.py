import numpy as np 

from dataset_wrappers.image_datasets import Cifar10Dataset, TinyCifar10Dataset
# from dataset_wrappers.feature_vector_datasets import ImageVectorizer
#from dataset_wrappers.feature_vector_datasets import HDF5FeatureVectorDataset
import numpy as np 
from mini_batch import MiniBatchGenerator, MiniBatchTrainer 
from transformations import *


from keras.models import Sequential
from keras.regularizers import l2
from keras.optimizers import SGD
from keras.layers import Dense, Convolution2D, Activation, MaxPooling2D, Flatten
from keras.utils import np_utils
import keras 



cifar10_classnames = {
											0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 
											5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'
										}


print "Loading Cifar10Datasets ..."
train_set = Cifar10Dataset('datasets/cifar-10-batches-py', "train")
train_set.load()
val_set = Cifar10Dataset('datasets/cifar-10-batches-py', "val")
val_set.load()
test_set = Cifar10Dataset('datasets/cifar-10-batches-py', "test")
test_set.load()


print "Setting up preprocessing transformation sequence ..."
preprocessing_transformation = TransformationSequence()

preprocessing_transformation.add_transformation(FloatCastTransformation())
print "Added FloatCastTransformation"

sub_trans = PerChannelSubtractionImageTransformation().from_dataset_mean(train_set)
print "Adding PerChannelSubtractionImageTransformation [train] (values: {})".format(sub_trans.values())
preprocessing_transformation.add_transformation(sub_trans)

div_trans = PerChannelDivisionImageTransformation().from_dataset_stddev(train_set)
print "Adding PerChannelDivisionImageTransformation [train] (values: {})".format(div_trans.values())
preprocessing_transformation.add_transformation(div_trans)

print "Appling transformations to train, val and test set ..."
preprocessing_transformation.apply_to_dataset(train_set)
preprocessing_transformation.apply_to_dataset(val_set)
preprocessing_transformation.apply_to_dataset(test_set)


print "Initializing minibatch generators with preprocessed datasets..."
train_batch_generator = MiniBatchGenerator(train_set, 64)
val_batch_generator = MiniBatchGenerator(val_set, 100)

print "Train model ..."
model = Sequential([
	Convolution2D(16,3,3, input_shape=(32,32,3), activation="relu", border_mode="same", W_regularizer=l2(0.0001)),
	MaxPooling2D(strides=(2,2)),
	Convolution2D(32,3,3, activation="relu", border_mode="same", W_regularizer=l2(0.0001)),
	MaxPooling2D(strides=(2,2)),
	Convolution2D(32,3,3, activation="relu", border_mode="same", W_regularizer=l2(0.0001)),
	MaxPooling2D(strides=(2,2)),
	Flatten(),
	Dense(10, activation="softmax", W_regularizer=l2(0.0001))
									])

model.compile(loss = "categorical_crossentropy", optimizer=SGD(lr=0.001,momentum =0.9, nesterov = True), metrics =["accuracy"])
model.summary()
acc, model = MiniBatchTrainer(model, train_batch_generator, val_batch_generator, early_stopping=10, verbose=True).train()

model.save("cnn_model.h5")




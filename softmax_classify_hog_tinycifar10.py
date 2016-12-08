import numpy as np 

# from dataset_wrappers.image_datasets import TinyCifar10Dataset
# from dataset_wrappers.feature_vector_datasets import ImageVectorizer
from dataset_wrappers.feature_vector_datasets import HDF5FeatureVectorDataset
import numpy as np 
from mini_batch import MiniBatchGenerator, MiniBatchTrainer 
from transformations import *


from keras.models import Sequential
from keras.regularizers import l2
from keras.optimizers import SGD
from keras.layers import Dense, Activation
from keras.utils import np_utils
import keras 



cifar10_classnames = {
											0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 
											5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'
										}


print "Loading HDF5FeatureVectorDatasets ..."
train_set = HDF5FeatureVectorDataset('datasets/features_tinycifar10_train.h5', cifar10_classnames)
val_set = HDF5FeatureVectorDataset('datasets/features_tinycifar10_val.h5', cifar10_classnames)
test_set = HDF5FeatureVectorDataset('datasets/features_tinycifar10_test.h5', cifar10_classnames)


print "Setting up preprocessing transformation sequence ..."
preprocessing_transformation = TransformationSequence()

preprocessing_transformation.add_transformation(FloatCastTransformation())
print "Added FloatCastTransformation"

sub_trans = SubtractionTransformation(0).from_dataset_mean(train_set)
print "Adding SubtractionTransformation [train] (value: {})".format(sub_trans.value())
preprocessing_transformation.add_transformation(sub_trans)

div_trans = DivisionTransformation(1).from_dataset_stddev(train_set)
print "Adding DivisionTransformation [train] (value: {})".format(div_trans.value())
preprocessing_transformation.add_transformation(div_trans)

print "Appling transformations to train, val and test set ..."
preprocessing_transformation.apply_to_dataset(train_set)
preprocessing_transformation.apply_to_dataset(val_set)
preprocessing_transformation.apply_to_dataset(test_set)


print "Initializing minibatch generators with preprocessed datasets..."
train_batch_generator = MiniBatchGenerator(train_set, 64)
val_batch_generator = MiniBatchGenerator(val_set, 100)

print "Train model ..."
model = Sequential([Dense(output_dim = 10, input_dim = 144), Activation("softmax")])
model.compile(loss = "categorical_crossentropy", optimizer=SGD(lr=0.001,momentum =0.9, nesterov = True), metrics =["accuracy"])
acc, model = MiniBatchTrainer(model, train_batch_generator, val_batch_generator, early_stopping=25, verbose=True).train()







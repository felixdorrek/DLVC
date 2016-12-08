import tensorflow as tf 
import numpy as np 

from dataset_wrappers.image_datasets import TinyCifar10Dataset
from dataset_wrappers.feature_vector_datasets import ImageVectorizer
import numpy as np 
from mini_batch import MiniBatchGenerator, MiniBatchTrainer 
from transformations import *

from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Dense, Activation
from keras.utils import np_utils
import keras 


print "Loading HDF5FeatureVectorDatasets ..."
train_set = ImageVectorizer(TinyCifar10Dataset("datasets/cifar-10-batches-py", "train"))
val_set = ImageVectorizer(TinyCifar10Dataset("datasets/cifar-10-batches-py", "val"))


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



print "Initializing minibatch generators with preprocessed datasets..."
train_batch_generator = MiniBatchGenerator(train_set, 64)
val_batch_generator = MiniBatchGenerator(val_set, 100)


print "Train model ..."
model = Sequential([Dense(output_dim = 10, input_dim = 3072), Activation("softmax")])
model.compile(loss = "categorical_crossentropy", optimizer=SGD(momentum =0.9, nesterov = True), metrics =["accuracy"])
MiniBatchTrainer(model, train_batch_generator, val_batch_generator).train()


## Pure tensorflow implementation
#from standard_tf_classifier import StandardTfClassifier 
#from linear_model import LinearModel 
#from optimizers import MomentumOptimizer
#sess = tf.Session()
#classifier = StandardTfClassifier(train_batch_generator, val_batch_generator, LinearModel(), MomentumOptimizer())
#classifier.train()







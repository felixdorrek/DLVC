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

print "Performing grid hyperparameter search ..."

learn_rates = [0.001,0.0008,0.006, 0.0004]

regularizations = [0.05, 0.02, 0.01, 0.007, 0.005, 0.001]

best_loss = 0
best_reg = 0
highest_accuracy = 0
best_model = 0

for l in learn_rates:
	for r in regularizations:
		print "Testing combination learn_rate={0}, regularization={1} ...".format(l, r)
		model = Sequential([
			Dense(output_dim = 100, input_dim = 144, W_regularizer=l2(r), activation="relu"),
			Dense(10, W_regularizer=l2(r), activation="softmax")
												])
		model.compile(loss = "categorical_crossentropy", optimizer=SGD(lr=l,momentum =0.9, nesterov = True), metrics =["accuracy"])
		acc, model = MiniBatchTrainer(model, train_batch_generator, val_batch_generator, early_stopping=25, verbose=False).train()
		if acc >= highest_accuracy:
			highest_accuracy = acc 
			best_model = model 

			best_learn = l
			best_reg = r 

best_model.save("softmax_classify_hog_best_model.h5")

print "Choosing best combination learn_rate={0}, regularization={1}".format(best_learn, best_reg)

print "Evaluating on test set ..."

acc = best_model.evaluate(test_set.get_raw_samples(), np_utils.to_categorical(test_set.get_labels()), batch_size=1000)

print "Test Accuracy: {0}".format(acc[1])
import math
import numpy as np
import os
import keras
from keras.utils import np_utils


class MiniBatchGenerator:
	  # Create minibatches of a given size from a dataset.
    # Preserves the original sample order unless shuffle() is used.

    def __init__(self, dataset, bs, tform=None):
    	self.dataset = dataset
    	self.bs = bs
    	self.tform = tform
    	self.batches_inds = np.split( 
    										np.arange( dataset.size() ),
    										[(bs)*(j+1) for j in range(self.nbatches()-1)]
    															)

    def get_dataset(self):
        return self.dataset

    def batchsize(self):
    	return self.bs

    def nbatches(self):
    	return int(math.ceil(float(self.dataset.size())/float(self.batchsize())))

    def shuffle(self):
    	self.batches_inds = np.split(
    										np.random.permutation(self.dataset.size()),
    										[(self.batchsize())*(j+1) for j in range(self.nbatches()-1)] 
    															)

    def batch(self, bid):
    	batch_inds = self.batches_inds[bid]
    	sample_batch = np.array(map(lambda j: self.dataset.sample(j)["sample"], batch_inds))
    	label_batch = np.array(map(lambda j: self.dataset.sample(j)["class_id"], batch_inds))
    	return {"samples":sample_batch, "labels":label_batch, "ids":batch_inds}


class MiniBatchTrainer:

    ## Trains a Keras model.

    def __init__(self, model, train_generator, val_generator, epochs=200, early_stopping=False, verbose = True):
        
        self.model = model
        self.train_generator = train_generator
        self.val_generator = val_generator
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.verb = verbose



    def train(self):
        train_generator = self.train_generator
        val_generator = self.val_generator

        nclasses = train_generator.get_dataset().nclasses()
        train_bs = train_generator.batchsize()
        val_bs = val_generator.batchsize()

        # For early stopping
        epochs_not_improved = 0
        highest_accuracy = 0

        for j in range(self.epochs):

            losses = np.array([])
            train_accs = np.array([])
            val_accs = np.array([])

            for i in range(train_generator.nbatches()):
                batch = train_generator.batch(i)
                samples = batch["samples"]
                labels = batch["labels"]

                self.model.train_on_batch(samples, np_utils.to_categorical(labels, nb_classes = nclasses))
                loss_and_metrics = self.model.evaluate(samples, np_utils.to_categorical(labels, nb_classes = nclasses), batch_size=train_bs, verbose=0)
                losses = np.append(losses, loss_and_metrics[0])
                train_accs = np.append(train_accs,loss_and_metrics[1])

            for i in range(val_generator.nbatches()):
                batch = val_generator.batch(i)
                samples = batch["samples"]
                labels = batch["labels"]
                loss_and_metrics = self.model.evaluate(samples, np_utils.to_categorical(labels, nb_classes = nclasses), batch_size=val_bs, verbose=0)
                val_accs = np.append(val_accs,loss_and_metrics[1])


            train_generator.shuffle()
            if self.verb:
                print "Epoch [{3}]: Train loss: {0:.3f}, Train accuracy: {1:.3f}, Validation accuracy: {2:.3f}".format(losses.mean(), train_accs.mean(), val_accs.mean(), j)


            if self.early_stopping:
                if highest_accuracy <= val_accs.mean():
                    highest_accuracy = val_accs.mean()
                    epochs_not_improved = 0
                    self.model.save("minibatch_trainer_tmp.h5")
                else:
                    epochs_not_improved += 1

                if epochs_not_improved == self.early_stopping:
                    if self.verb:
                        print "No improvement for {} epochs ... early stopping".format(self.early_stopping)
                    print "Validation accuracy: {0} (Epoch {1})".format(highest_accuracy, j)

                    best_model = keras.models.load_model("minibatch_trainer_tmp.h5")
                    os.remove("minibatch_trainer_tmp.h5")
                    return highest_accuracy, best_model

        if self.early_stopping:
            best_model = keras.models.load_model("minibatch_trainer_tmp.h5")
            os.remove("minibatch_trainer_tmp.h5")
            return highest_accuracy, best_model

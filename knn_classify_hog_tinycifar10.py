from knn_classifier import KnnClassifier
from dataset_wrappers.feature_vector_datasets import HDF5FeatureVectorDataset
import numpy as np 
import time
from collections import Counter
import matplotlib.pyplot as plt 

ks = range(1,41)
cmps = ["l1", "l2"]				

knn = KnnClassifier(1, "l2")

cifar10_classnames = {
											0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 
											5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'
										}

train_dataset = HDF5FeatureVectorDataset('datasets/features_tinycifar10_train.h5', cifar10_classnames)
val_dataset = HDF5FeatureVectorDataset('datasets/features_tinycifar10_val.h5', cifar10_classnames)
test_dataset = HDF5FeatureVectorDataset('datasets/features_tinycifar10_test.h5', cifar10_classnames)


knn.train(train_dataset)

#print knn.score(val_dataset)
#print knn.score(test_dataset)


print "Performing random hyperparameter search ..."
print "[train] {0} samples".format(train_dataset.size())
print "[val] {0} samples".format(val_dataset.size())

param_grid = [(k,"l1") for k in ks] + [(k,"l2") for k in ks]
best = knn 
best_comb = (1, "l2")
best_accuracy = 0.308

# ks_l1 = []
# acc_l1 = []
# ks_l2 = [1]
# acc_l2 = [0.308]

for j in range(20):
	start_time = time.time()
	ind = np.random.choice(len(param_grid))
	k, cmp = param_grid[ind]
	param_grid.remove((k, cmp))
	knn = KnnClassifier(k, cmp)
	knn.train(train_dataset)
	print "k={0}, cmp = {1}:".format(k, cmp)
	accuracy = knn.score(val_dataset)
	# if (cmp == "l1"):
	# 	ks_l1+= [k] 
	# 	acc_l1 += [accuracy]
	# else:
	#  	ks_l2 += [k]
	#  	acc_l2 += [accuracy]
	if accuracy > best_accuracy:
		best_accuracy = accuracy
		best_comb = (k, cmp)
		best = knn
	print "Accuracy: {0}%".format(accuracy*100)
	print("--- {0} seconds ---".format(time.time() - start_time))



print "Testing best combination {0} on test set ...".format(best_comb)
print "[Test] {0} samples".format(test_dataset.size())
print "Accuracy: {0}%".format(best.score(test_dataset)*100)


# plt.ylabel("Accuracy")
# plt.xlabel("k")
# plt.xlim(0,41)
# plt.ylim(0.28, 0.50)
# plt.plot(ks_l1, acc_l1, "bx", label= "dist = l1")
# plt.plot(ks_l2, acc_l2, "gx", label= "dist = l2")
# plt.plot(best_comb[0], best_accuracy, "rx", label = "dist = {0}".format(best_comb[1]))
# plt.legend()
# plt.show()
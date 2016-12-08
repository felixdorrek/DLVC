import tensorflow as tf 
import numpy as np 
import time


class StandardTfClassifier():

	def __init__(self, train_data_generator, val_data_generator, inference_model, optimizer):

		self.train_data_generator = train_data_generator
		self.val_data_generator = val_data_generator
		self.inference_model = inference_model
		self.optimizer = optimizer


	def train(self, epochs=200):

		train_data_generator = self.train_data_generator
		val_data_generator = self.val_data_generator

		# Build the Tensorflow computation graph
		#TODO .rows
		samples = tf.placeholder(tf.float32, shape=[None, 3072])
		labels = tf.placeholder(tf.int32, shape=[None])

		logits = self.inference_model.inference(samples)

		loss = self.loss(logits, labels)

		evaluation = self.evaluate(logits, labels)

		train_step = self.optimizer.train_step(loss) 

		init = tf.initialize_all_variables()

		# Start Session and initialize all Variables
		sess = tf.Session()
		sess.run(init)


		# Run the training 
		for i in range(epochs):
			losses = np.array([])
			train_accs = np.array([])
			val_accs = np.array([])

			for j in range(train_data_generator.nbatches()): 
				batch = train_data_generator.batch(j)
				samples_batch = batch["samples"]
				labels_batch = batch["labels"]
				_, loss_value = sess.run([train_step, loss], feed_dict={samples: samples_batch, labels: labels_batch})
				losses = np.append(losses, loss_value)
				train_accs = np.append(
										train_accs, 
										sess.run( 
											evaluation,
											feed_dict={samples: samples_batch, labels: labels_batch}
														)
															)

			for j in range(val_data_generator.nbatches()):
				val_batch = val_data_generator.batch(j)
				samples_batch = val_batch["samples"]
				samples_batch.shape
				labels_batch = val_batch["labels"]
				val_accs = np.append(
						val_accs, 
						sess.run( 
							evaluation,
							feed_dict={samples: samples_batch, labels: labels_batch}
										)
														)

			train_data_generator.shuffle()
			print "Epoch [{3}]: Train loss: {0:.3f}, Train accuracy: {1:.3f}, Validation accuracy: {2:.3f}".format(losses.mean(), train_accs.mean(), val_accs.mean(), i)



	def loss(self, logits, labels):
		return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels))


	def evaluate(self, logits, labels): 
	  correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(tf.one_hot(labels,10), 1))
	  return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))





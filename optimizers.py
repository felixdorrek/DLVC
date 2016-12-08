import tensorflow as tf 


class MomentumOptimizer():

	def train_step(self, loss):
		return tf.train.MomentumOptimizer(0.01,0.9).minimize(loss)


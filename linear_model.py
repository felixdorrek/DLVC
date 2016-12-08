import tensorflow as tf 

class LinearModel():

	def inference(self, features):		
		W = self._weight_variable([3072, 10])
		b = self._bias_variable([10])
		return tf.matmul(features,W) + b

	def _weight_variable(self, shape):
		initial = tf.truncated_normal(shape, stddev=0.1)
		return tf.Variable(initial)

	def _bias_variable(self, shape):
		initial = tf.constant(0.1, shape=shape)
		return tf.Variable(initial)
from dataset_wrappers.image_datasets import TinyCifar10Dataset
from transformations import *


dataset = TinyCifar10Dataset("datasets/cifar-10-batches-py", "train")
dataset.load()

subtraction_transformation = SubtractionTransformation(0).from_dataset_mean(dataset)
division_transformation = DivisionTransformation(1).from_dataset_stddev(dataset)


print "Computing SubtractionTransformation from TinyCifar10Dataset [{0}] mean".format(dataset.split)
print " Value {0}".format(subtraction_transformation.value())

print "Computing DivisionTransformation from TinyCifar10Dataset [{0}] stddev".format(dataset.split)
print " Value {0}".format(division_transformation.value())

sample = dataset.sample(0)["sample"]

print (
				"First sample of TinyCifar10Dataset [{0}]: ".format(dataset.split) + 
				"shape: {0}, data type: {1}, mean: {2}, min: {3}, max: {4}".format(
																																						sample.shape,
																																						sample.dtype,
																																						sample.mean(),
																																						sample.min(),
																																						sample.max()
																																						 )
			)

sample = IdentityTransformation().apply(sample)

print (
				"After applying IdentityTransformation:" + 
				"shape: {0}, data type: {1}, mean: {2}, min: {3}, max: {4}".format(
																																						sample.shape,
																																						sample.dtype,
																																						sample.mean(),
																																						sample.min(),
																																						sample.max()
																																						 )
			)

sample_cast = FloatCastTransformation().apply(sample)

print (
				"After applying FloatCastTransformation:" + 
				"shape: {0}, data type: {1}, mean: {2}, min: {3}, max: {4}".format(
																																						sample_cast.shape,
																																						sample_cast.dtype,
																																						sample_cast.mean(),
																																						sample_cast.min(),
																																						sample_cast.max()
																																						 )
			)


sequence_one = TransformationSequence()
sequence_one.add_transformation(FloatCastTransformation())
sequence_one.add_transformation(subtraction_transformation)
sample_seq1 = sequence_one.apply(sample)

print (
				"After applying sequence FloatCast -> SubtractionTransformation:" + 
				"shape: {0}, data type: {1}, mean: {2}, min: {3}, max: {4}".format(
																																						sample_seq1.shape,
																																						sample_seq1.dtype,
																																						sample_seq1.mean(),
																																						sample_seq1.min(),
																																						sample_seq1.max()
																																						 )
			)

sequence_two = TransformationSequence([FloatCastTransformation(), subtraction_transformation, division_transformation])
sample_seq2 = sequence_two.apply(sample)

print (
				"After applying sequence FloatCast -> SubtractionTransformation -> DivisionTransformation:" + 
				"shape: {0}, data type: {1}, mean: {2}, min: {3}, max: {4}".format(
																																						sample_seq2.shape,
																																						sample_seq2.dtype,
																																						sample_seq2.mean(),
																																						sample_seq2.min(),
																																						sample_seq2.max()
																																						 )
			)

print "Computing PerChannelSubtractionImageTransformation from TinyCifar10Dataset [train] mean"

per_channel_mean = PerChannelSubtractionImageTransformation().from_dataset_mean(dataset)
vals = per_channel_mean.values
print "Values: {0},{1},{2}".format(vals[0], vals[1], vals[2])


print "Computing PerChannelDivisionImageTransformation from TinyCifar10Dataset [train] stddev"

per_channel_std = PerChannelDivisionImageTransformation().from_dataset_stddev(dataset)
vals = per_channel_std.values
print "Values: {0},{1},{2}".format(vals[0], vals[1], vals[2])

per_channel_sequence = TransformationSequence([FloatCastTransformation(), per_channel_mean, per_channel_std])
print "After applying sequence FloatCast -> PerChannelSubtractionImageTransformation -> PerChannelDivisionImageTransformation:"
per_channel_sample = per_channel_sequence.apply(sample)

print"shape: {0}, data type: {1}, mean: {2}, min: {3}, max: {4}".format(per_channel_sample.shape, per_channel_sample.dtype, per_channel_sample.mean(), per_channel_sample.min(), per_channel_sample.max())

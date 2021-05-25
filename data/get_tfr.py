import numpy as np
import tensorflow.compat.v1 as tf
tf.enable_eager_execution()

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))): # if value ist tensor
        value = value.numpy() # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_array(array):
  array = tf.io.serialize_tensor(array)
  return array

arrays = np.load('flickr/feats.npy')

file_path = 'big_feats.tfrecords'
with tf.io.TFRecordWriter(file_path) as writer:
  for array in arrays:
    serialized_array = serialize_array(array)
    feature = {'b_feature': _bytes_feature(serialized_array)}
    example_message = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example_message.SerializeToString())
print('Done!')

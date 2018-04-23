import random
import tensorflow as tf
import math
import os
import sys

slim = tf.contrib.slim

#State the labels filename
LABELS_FILENAME = 'labels.txt'

#===============DEFINE YOUR ARGUMENTS==============
flags = tf.app.flags

#State your dataset file
flags.DEFINE_string('dataset_file', "data.txt", 'String: Your dataset txt file')

# The number of shards per dataset split.
flags.DEFINE_integer('num_shards', 2, 'Int: Number of shards to split the TFRecord files')

# The number of images in the validation set. You would have to know the total number of examples in advance. This is essentially your evaluation dataset.
flags.DEFINE_float('validation_size', 0.2, 'Float: The proportion of examples in the dataset to be used for validation')
# Seed for repeatability.
flags.DEFINE_integer('random_seed', 0, 'Int: Random seed to use for repeatability.')

#Output filename for the naming the TFRecord file
flags.DEFINE_string('tfrecord_filename', "fundus", 'String: The output filename to name your TFRecord file')

FLAGS = flags.FLAGS
#============================Dataset Utils ===================================================

def int64_feature(values):
  """Returns a TF-Feature of int64s.
  Args:
    values: A scalar or list of values.
  Returns:
    a TF-Feature.
  """
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def bytes_feature(values):
  """Returns a TF-Feature of bytes.
  Args:
    values: A string.
  Returns:
    a TF-Feature.
  """
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def image_to_tfexample(image_data, image_format, height, width, class_id):
  return tf.train.Example(features=tf.train.Features(feature={
      'image/encoded': bytes_feature(image_data),
      'image/format': bytes_feature(image_format),
      'image/class/label': int64_feature(class_id),
      'image/height': int64_feature(height),
      'image/width': int64_feature(width),
  }))

def write_label_file(labels_to_class_names, dataset_file,
                     filename=LABELS_FILENAME):
  """Writes a file with the list of class names.
  Args:
    labels_to_class_names: A map of (integer) labels to class names.
    dataset_file: The directory in which the labels file should be written.
    filename: The filename where the class names are written.
  """
  labels_filename = os.path.join(dataset_file, filename)
  with tf.gfile.Open(labels_filename, 'w') as f:
    for label in labels_to_class_names:
      class_name = labels_to_class_names[label]
      f.write('%d:%s\n' % (label, class_name))


def has_labels(dataset_file, filename=LABELS_FILENAME):
  """Specifies whether or not the dataset directory contains a label map file.
  Args:
    dataset_file: The directory in which the labels file is found.
filename: The filename where the class names are written.
  Returns:
    `True` if the labels file exists and `False` otherwise.
  """
  return tf.gfile.Exists(os.path.join(dataset_file, filename))


def read_label_file(dataset_file, filename=LABELS_FILENAME):
  """Reads the labels file and returns a mapping from ID to class name.
  Args:
    dataset_file: The directory in which the labels file is found.
    filename: The filename where the class names are written.
  Returns:
    A map from a label (integer) to class name.
  """
  labels_filename = os.path.join(dataset_file, filename)
  with tf.gfile.Open(labels_filename, 'r') as f:
    lines = f.read().decode()
  lines = lines.split('\n')
  lines = filter(None, lines)

  labels_to_class_names = {}
  for line in lines:
    index = line.index(':')
    labels_to_class_names[int(line[:index])] = line[index+1:]
  return labels_to_class_names

#=======================================  Conversion Utils  ===================================================

#Create an image reader object for easy reading of the images
class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Initializes function that decodes RGB jpeg data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def read_image_dims(self, sess, image_data):
    image = self.decode_jpeg(sess, image_data)
                                                     image = self.decode_jpeg(sess, image_data)
    return image.shape[0], image.shape[1]

  def decode_jpeg(self, sess, image_data):
    image = sess.run(self._decode_jpeg,
                     feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image


def _get_filenames_and_classes(dataset_file):
  """Returns a list of filenames and inferred class names.
  Args:
    dataset_file: A directory containing a set of subdirectories representing
      class names. Each subdirectory should contain jpeg or jpeg encoded images.
  Returns:
    A list of image file paths, relative to `dataset_file` and the list of
    subdirectories, representing class names.
  """
  files = [line.rstrip() for line in open(dataset_file)]
  photo_filenames = [filename.split(" ")[1] for filename in files]
  class_names = [filename.split(" ")[0] for filename in files]

  return photo_filenames,class_names


def _get_dataset_filename(dataset_file, split_name, shard_id, tfrecord_filename, _NUM_SHARDS):
  output_filename = '%s_%s_%05d-of-%05d.tfrecord' % (
      tfrecord_filename, split_name, shard_id, _NUM_SHARDS)
  return output_filename


def _convert_dataset(split_name, filenames, class_ids, dataset_file, tfrecord_filename, _NUM_SHARDS):
  """Converts the given filenames to a TFRecord dataset.
  Args:
    split_name: The name of the dataset, either 'train' or 'validation'.
    filenames: A list of absolute paths to jpeg or jpeg images.
    class_names_to_ids: A dictionary from class names (strings) to ids
      (integers).
    dataset_file: The directory where the converted datasets are stored.
                                                                                                                             120,3         58%
 """
  assert split_name in ['train', 'validation']
  
  num_per_shard = int(math.ceil(len(filenames) / float(_NUM_SHARDS)))

  with tf.Graph().as_default():
    image_reader = ImageReader()
    
    with tf.Session('') as sess:

      for shard_id in range(_NUM_SHARDS):
        output_filename = _get_dataset_filename(
            dataset_file, split_name, shard_id, tfrecord_filename = tfrecord_filename, _NUM_SHARDS = _NUM_SHARDS)

        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
          start_ndx = shard_id * num_per_shard
          end_ndx = min((shard_id+1) * num_per_shard, len(filenames))
          for i in range(start_ndx, end_ndx):
            sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                i+1, len(filenames), shard_id))
            sys.stdout.flush()
  
            # Read the filename:
            image_data = tf.gfile.FastGFile(filenames[i], 'rb').read()
            height, width = image_reader.read_image_dims(sess, image_data)

            class_id = class_ids[i]
  
            example = image_to_tfexample(
                image_data, b'jpeg', height, width, class_id)
            tfrecord_writer.write(example.SerializeToString())

  sys.stdout.write('\n')
  sys.stdout.flush()

def _dataset_exists(dataset_file, _NUM_SHARDS, output_filename):
  for split_name in ['train', 'validation']:
    for shard_id in range(_NUM_SHARDS):
      tfrecord_filename = _get_dataset_filename(
          dataset_file, split_name, shard_id, output_filename, _NUM_SHARDS)
 if not tf.gfile.Exists(tfrecord_filename):
        return False
  return True
def main():

    #=============CHECKS==============
    #Check if there is a tfrecord_filename entered
    if not FLAGS.tfrecord_filename:
        raise ValueError('tfrecord_filename is empty. Please state a tfrecord_filename argument.')

    #If the TFRecord files already exist in the directory, then exit without creating the files again
    if _dataset_exists(dataset_file = FLAGS.dataset_file, _NUM_SHARDS = FLAGS.num_shards, output_filename = FLAGS.tfrecord_filename):
        print('Dataset files already exist. Exiting without re-creating them.')
        return None
    #==========END OF CHECKS============

    #Get a list of photo_filenames like ['123.jpg', '456.jpg'...] and a list of sorted class names from parsing the subdirectories.
    photo_filenames, class_names = _get_filenames_and_classes(FLAGS.dataset_file)

    #Refer each of the class name to a specific integer number for predictions later
    class_ids = [1 if label == "1" else 0 for label in class_names]

    #Find the number of validation examples we need
    num_validation = int(FLAGS.validation_size * len(photo_filenames))

    training_filenames = photo_filenames
    validation_filenames = photo_filenames[:num_validation]
    training_labels = class_ids
    validation_labels = class_ids[:num_validation]
    # First, convert the training and validation sets.
    _convert_dataset('train', training_filenames, training_labels,
                     dataset_file = FLAGS.dataset_file,
                     tfrecord_filename = FLAGS.tfrecord_filename,
                     _NUM_SHARDS = FLAGS.num_shards)
    _convert_dataset('validation', validation_filenames, validation_labels,
                     dataset_file = FLAGS.dataset_file,
                     tfrecord_filename = FLAGS.tfrecord_filename,
                     _NUM_SHARDS = FLAGS.num_shards)

    print('\nFinished converting the %s dataset!' % (FLAGS.tfrecord_filename))



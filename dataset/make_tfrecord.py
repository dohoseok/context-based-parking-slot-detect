# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Converts parking_space data to TFRecords of TF-Example protos.

This module downloads the Flowers data, uncompresses it, reads the files
that make up the Flowers data and creates two TFRecord datasets: one for train
and one for test. Each TFRecord dataset is comprised of a set of TF-Example
protocol buffers, each of which contain a single image and label.

The script should take about a minute to run.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import platform
import tensorflow as tf

from dataset import tf_dataset_utils

# The number of images in the validation set. 20%
_NUM_VALIDATION = 0

# Seed for repeatability.
_RANDOM_SEED = 0

# The number of shards per dataset split.
# _NUM_SHARDS = 50

_BYTE_PER_TFRECORD = 40*(2**20)

class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def read_image_dims(self, sess, image_data):
    image = self.decode_jpeg(sess, image_data)
    return image.shape[0], image.shape[1]

  def decode_jpeg(self, sess, image_data):
    image = sess.run(self._decode_jpeg,
                     feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image


def _get_filenames_and_classes(dataset_dir):
  """Returns a list of filenames and inferred class names.

  Args:
    dataset_dir: A directory containing a set of subdirectories representing
      class names. Each subdirectory should contain PNG or JPG encoded images.

  Returns:
    A list of image file paths, relative to `dataset_dir` and the list of
    subdirectories, representing class names.
  """
  parking_space_root = os.path.join(dataset_dir, 'image')
  directories = []
  class_names = []
  for filename in os.listdir(parking_space_root):
    path = os.path.join(parking_space_root, filename)
    if os.path.isdir(path):
      directories.append(path)
      class_names.append(filename)

  photo_filenames = []
  for directory in directories:
    for filename in os.listdir(directory):
      path = os.path.join(directory, filename)
      photo_filenames.append(path)

  return photo_filenames, sorted(class_names)


def _get_dataset_filename(dataset_dir, split_name, shard_id, num_tfrecord):
  output_filename = 'parking_%s_%05d-of-%05d.tfrecord' % (
      split_name, shard_id, num_tfrecord)
  return os.path.join(dataset_dir, output_filename)


def _convert_dataset(split_name, filenames, dataset_dir, num_tfrecord):
  """Converts the given filenames to a TFRecord dataset.

  Args:
    split_name: The name of the dataset, either 'train' or 'validation'.
    filenames: A list of absolute paths to png or jpg images.
    class_names_to_ids: A dictionary from class names (strings) to ids
      (integers).
    dataset_dir: The directory where the converted datasets are stored.
  """
  # assert split_name in ['train', 'validation']

  image_path = os.path.join(dataset_dir, "image")
  label_path = os.path.join(dataset_dir, "label")
  tfrecord_path = os.path.join(dataset_dir, "tfrecord")
  os.makedirs(tfrecord_path, exist_ok=True)

  num_per_shard = int(math.ceil(len(filenames) / float(num_tfrecord)))

  with tf.Graph().as_default():
    image_reader = ImageReader()

    with tf.Session('') as sess:

      for shard_id in range(num_tfrecord):
        output_filename = _get_dataset_filename(
            tfrecord_path, split_name, shard_id, num_tfrecord)

        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
          start_ndx = shard_id * num_per_shard
          end_ndx = min((shard_id+1) * num_per_shard, len(filenames))
          for i in range(start_ndx, end_ndx):
            sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                i+1, len(filenames), shard_id))
            sys.stdout.write(' '+ str(filenames[i])+'\n')
            sys.stdout.flush()

            full_jpg_name = os.path.join(image_path, filenames[i])
            full_txt_name = os.path.join(label_path, filenames[i]).replace(".jpg",".txt")

            # Read the filename:
            image_data = tf.gfile.FastGFile(full_jpg_name, 'rb').read()
            height, width = image_reader.read_image_dims(sess, image_data)

            f = open(full_txt_name)
            print(full_txt_name)
            lines = f.readlines()
            class_id = int(lines[0][0])
            angle = int(lines[1][:-1])
            print(filenames[i], class_id, angle)

            f.close()

            example = tf_dataset_utils.image_to_tfexample(
                image_data, b'jpg', height, width, class_id, angle, tf.compat.as_bytes(full_jpg_name))
            tfrecord_writer.write(example.SerializeToString())

  sys.stdout.write('\n')
  sys.stdout.flush()


def get_size(start_path='.'):
  total_size = 0
  for dirpath, dirnames, filenames in os.walk(start_path):
    for f in filenames:
      fp = os.path.join(dirpath, f)
      # skip if it is symbolic link
      if not os.path.islink(fp):
        total_size += os.path.getsize(fp)

  return total_size


def get_dir_size(path):
    total_size = 0
    if platform.system() == 'Windows':
        import win32file
        if os.path.isdir(path):
            items = win32file.FindFilesW(path + '\\*')# Add the size or perform recursion on folders.
            for item in items:
                size = item[5]
                total_size += size
    else:
        total_size = get_size(path)
    return total_size


def run(dataset_dir, type = 'train'):
  print(os.path.join(dataset_dir, 'image'))
  datasize = get_dir_size(os.path.join(dataset_dir, 'image'))
  num_tfrecord = datasize//_BYTE_PER_TFRECORD + 1
  print("JPG data size = ", datasize//(2**20), "MB, num_tfrecord =", num_tfrecord)
  """Runs the download and conversion operation.

  Args:
    dataset_dir: The dataset directory where the dataset is stored.
  """
  if not tf.gfile.Exists(dataset_dir):
    tf.gfile.MakeDirs(dataset_dir)

  photo_filenames = os.listdir(os.path.join(dataset_dir, 'image'))

  _convert_dataset(type, photo_filenames, dataset_dir, num_tfrecord)
  print('\nFinished converting the parking_space dataset!')

  return len(photo_filenames)
import tensorflow as tf

from parking_context_recognizer.config import *


def _parse_function(proto):
    # define your tfrecord again. Remember that you saved your image as a string.
    keys_to_features = {
        'image/class/label': tf.io.FixedLenFeature([], tf.int64),
        'image/angle': tf.io.FixedLenFeature([], tf.int64),
        'image/encoded': tf.io.FixedLenFeature([], tf.string)
    }

    # Load one example
    parsed_features = tf.io.parse_single_example(proto, keys_to_features)
    image = tf.image.decode_jpeg(parsed_features['image/encoded'], channels=3)
    image = tf.image.resize(image, [INPUT_HEIGHT, INPUT_WIDTH])
    image = tf.image.per_image_standardization(image)
    type = tf.cast(parsed_features['image/class/label'], tf.int32)
    angle = tf.cast(parsed_features['image/angle'], tf.float32)/180. + .5

    return image, type, angle


def create_dataset(filepath, is_test = False):
    # This works with arrays as well
    dataset = tf.data.TFRecordDataset(filepath)

    # Maps the parser on every filepath in the array. You can set the number of parallel loaders here
    dataset = dataset.map(_parse_function)

    if not is_test:
        # This dataset will go on forever
        dataset = dataset.repeat()

        # Set the number of datapoints you want to load and shuffle
        dataset = dataset.shuffle(SHUFFLE_BUFFER)
    else:
        dataset = dataset.repeat(1)

    # Set the batchsize
    dataset = dataset.batch(BATCH_SIZE)

    # Create an iterator
    iterator = dataset.make_one_shot_iterator()

    # Create your tf representation of the iterator
    image, type, angle = iterator.get_next()

    return image, type, angle
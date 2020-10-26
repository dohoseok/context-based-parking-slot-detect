import time
import warnings
import glob
import itertools
import os
import numpy as np
import tensorflow as tf

from parking_context_recognizer.utils import *
from parking_context_recognizer.models import model_mobilenetv2 as model


def train(train_path, pre_weight = None):
    warnings.filterwarnings("ignore")
    tf.logging.set_verbosity(tf.logging.ERROR)
    today = time.localtime()
    current_time = "%02d%02d%02d_%02d%02d" % (today.tm_year, today.tm_mon, today.tm_mday, today.tm_hour, today.tm_min)
    checkpoint_dir = os.path.join('weight_pcr', current_time)
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = checkpoint_dir + "/cp-{epoch:04d}.ckpt"

    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, verbose=1, save_weights_only=True, save_freq='epoch', period=10)

    def decay_schedule(epoch, lr):
        # decay by 0.1 every 5 epochs; use `% 1` to decay after each epoch
        if epoch in [20, 30, 40]:
            lr = lr * 0.1
        return lr

    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(decay_schedule)
    
    filenames_train = tf.data.Dataset.list_files(os.path.join(train_path, "train/tfrecord/") + "*.tfrecord")
    
	#Get your datatensors
    image, type, angle = create_dataset(filenames_train)

    model_input = tf.keras.layers.Input(tensor=image)
    train_model = model(model_input)

    train_model.summary()

    if not pre_weight is None:
        train_model.load_weights(os.path.join('weight_pcr', pre_weight))

    #Train the model
    train_model.fit(image, [type, angle], callbacks = [cp_callback, lr_scheduler], epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, use_multiprocessing=True)

    return checkpoint_dir


def evaluate(test_path, weight_file):
    warnings.filterwarnings("ignore")
    tf.logging.set_verbosity(tf.logging.ERROR)
    tfrecord_files = glob.glob(os.path.join(test_path,"tfrecord/") + "*.tfrecord")
    filenames = []
    types = []
    angles = []

    for tfrecord_file in tfrecord_files:
        record_iterator = tf.python_io.tf_record_iterator(tfrecord_file)
        for string_record in record_iterator:
            example = tf.train.Example()
            example.ParseFromString(string_record)

            filename = os.path.basename(str(example.features.feature['image/filename'].bytes_list.value[0]))[:-1]
            filenames.append(filename)
            types.append(example.features.feature['image/class/label'].int64_list.value[0])
            angles.append(example.features.feature['image/angle'].int64_list.value[0])

    image, type, angle = create_dataset(tfrecord_files, is_test=True)
    model_input = tf.keras.layers.Input(tensor=image)

    test_model = model(model_input)
    test_model.load_weights(weight_file)

    start_time = time.time()
    type_predict, angle_predict = test_model.predict(image, steps = int(len(filenames)/BATCH_SIZE + 1))
    print("time : ", time.time() - start_time)

    tf.keras.backend.clear_session()

    type_predict = np.argmax(type_predict, axis=1)
    type_predict = np.ndarray.tolist(type_predict)

    angle_predict = angle_predict * 180. - 90.
    angle_predict = list(itertools.chain.from_iterable(angle_predict))

    os.makedirs('result', exist_ok=True)
    f_0 = open(os.path.join('result', 'result_pcr_type_0.txt'), 'wt')
    f_1 = open(os.path.join('result', 'result_pcr_type_1.txt'), 'wt')
    f_2 = open(os.path.join('result', 'result_pcr_type_2.txt'), 'wt')
    f_list = [f_0, f_1, f_2]
    type_count = [0, 0, 0]
    for filename, type, angle in zip(filenames, type_predict, angle_predict):
        if type == 3:
            continue
        f = f_list[type]
        count = type_count[type]
        f.write('{} {} 256 768 {}\n'.format(count, os.path.join(os.path.join(test_path,"image/"), filename), int(round(angle, 0))))
        type_count[type] += 1
    for f in f_list:
        f.close()

    def diff_xy(x,y):
        if abs(x-y) > 180:
            return abs(abs(x-y)-360)
        else:
            return abs(x-y)

    accuracy = sum(1 for x,y in zip(types, type_predict) if x == y) / float(len(type_predict))
    angle_mae = sum((z!=3)*diff_xy(x,y) for x, y, z in zip(angles, angle_predict, types)) / float(sum(1 for x in zip(types) if x != 3))

    print(accuracy)
    print(angle_mae)

    return accuracy, angle_mae

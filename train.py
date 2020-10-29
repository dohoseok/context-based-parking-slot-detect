import argparse
import os

import parking_context_recognizer.train as pcr_train
import parking_slot_detector.train as psd_train

#################
# ArgumentParser
#################
parser = argparse.ArgumentParser(description="context-based parking slot detector")

parser.add_argument("--data_path", type=str, default="/home/mind3/project/dataset/PIL-park/",
                    help="The path of the train tfrecord file.")

args = parser.parse_args()

# Train Parking Context Recognizer
pcr_train.train(train_path=args.data_path)

# Train Parking Slot Detector
trained_path = psd_train.train(args.data_path, 'pre_weight/yolov3.ckpt', 'weight_psd')
psd_train.train(os.path.join(args.data_path, "t0"), trained_path, 'weight_psd/type_0', fine_tune=True)
psd_train.train(os.path.join(args.data_path, "t1"), trained_path, 'weight_psd/type_1', fine_tune=True)
psd_train.train(os.path.join(args.data_path, "t2"), trained_path, 'weight_psd/type_2', fine_tune=True)

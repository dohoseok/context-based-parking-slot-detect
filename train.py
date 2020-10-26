import argparse

import parking_context_recognizer.train as pcr_train
import parking_slot_detector.train as psd_train

#################
# ArgumentParser
#################
parser = argparse.ArgumentParser(description="context-based parking slot detector")

parser.add_argument("--pcr_train_path", type=str, default="/home/mind3/project/dataset/PIL-park/train/",
                    help="The path of the train tfrecord file.")
# parser.add_argument("--pcr_test_weight", type=str, default="weight_pcr/trained/trained.ckpt",
#                     help="The path of the weights to restore.")

parser.add_argument("--psd_train_path", type=str, default="/home/mind3/project/dataset/PIL-park/",
                    help="The path of the train tfrecord file.")

args = parser.parse_args()

# Train Parking Context Recognizer
pcr_train.train(train_path=args.pcr_train_path)

# Train Parking Slot Detector
trained_path = psd_train.train(args.psd_train_path, 'pre_weight/yolov3.ckpt', 'weight_psd')
psd_train.train(os.path.join(args.psd_train_path, "t0"), trained_path, 'weight_psd/type_0', fine_tune=True)
psd_train.train(os.path.join(args.psd_train_path, "t1"), trained_path, 'weight_psd/type_1', fine_tune=True)
psd_train.train(os.path.join(args.psd_train_path, "t2"), trained_path, 'weight_psd/type_2', fine_tune=True)

import argparse

import parking_context_recognizer.train as pcr_train

#################
# ArgumentParser
#################
parser = argparse.ArgumentParser(description="context-based parking slot detector")

parser.add_argument("--pcr_train_path", type=str, default="E:/parking_space/data_psd_04_1/train",
                    help="The path of the train tfrecord file.")
# parser.add_argument("--pcr_test_weight", type=str, default="weight_pcr/trained/trained.ckpt",
#                     help="The path of the weights to restore.")

args = parser.parse_args()

pcr_train.train_run(train_path=args.pcr_train_path)
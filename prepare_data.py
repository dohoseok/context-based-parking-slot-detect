import argparse
import os

from dataset import make_tfrecord, dataset_utils, data_augmentation

#################
# ArgumentParser
#################
parser = argparse.ArgumentParser(description="context-based parking slot detector")

parser.add_argument("--data_path", type=str, default="E:/PIL-park",
                    help="The path of the parking slot detection dataset.")

args = parser.parse_args()

make_tfrecord.run(os.path.join(args.data_path, "test"), 'test')
dataset_utils.create_detector_data_file(os.path.join(args.data_path, "test"), os.path.join(args.data_path, "test.txt"))

# data_augmentation.run(os.path.join(args.data_path, "train"))
# make_tfrecord.run(os.path.join(args.data_path, "train"), 'train')
# make_detector_data.create_detector_data_file(os.path.join(args.data_path, "train"), os.path.join(args.data_path, "train.txt"), include_non_park=True)
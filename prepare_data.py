import argparse
import os

from dataset import make_tfrecord, dataset_utils, data_augmentation

#################
# ArgumentParser
#################
parser = argparse.ArgumentParser(description="context-based parking slot detector")

parser.add_argument("--data_path", type=str, default="/home/mind3/project/dataset/PIL-park",
                    help="The path of the parking slot detection dataset.")

args = parser.parse_args()

# Prepare data for PCR test
make_tfrecord.run(os.path.join(args.data_path, "test"), 'test')
# Prepare data for PSD test
dataset_utils.create_detector_data_file(os.path.join(args.data_path, "test"), os.path.join(args.data_path, "test.txt"), include_non_park=True)

# Data augmentation for train
data_augmentation.run(os.path.join(args.data_path, "train"))
# Prepare data for PCR train
make_tfrecord.run(os.path.join(args.data_path, "train"), 'train')
# Prepare data for PSD train
dataset_utils.create_detector_data_file(os.path.join(args.data_path, "train"), os.path.join(args.data_path, "train.txt"), include_non_park=True)
# Prepare data for PSD train (Fine-tune)
dataset_utils.create_detector_data_file_fine_tune(args.data_path)

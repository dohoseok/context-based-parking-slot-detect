import argparse
import os

from parking_context_recognizer import train as pcr_train
from parking_slot_detector import test as psd_test
from parking_slot_detector import merge_three_type_result_files as merge_result
from parking_slot_detector.utils import eval_utils as eval_utils

#################
# ArgumentParser
#################
parser = argparse.ArgumentParser(description="context-based parking slot detector")

parser.add_argument("--data_path", type=str, default="E:/PIL-park/",
                    help="The path of the parking slot detection dataset")
parser.add_argument("--pcr_test_weight", type=str, default="weight_pcr/trained/trained.ckpt",
                    help="The path of the trained weights of pcr.")

parser.add_argument("--psd_test_weight_type0", type=str, default="weight_psd/fine_tuned_type_0",
                    help="The path of the trained weights of fine-tuned to parallel type.")

parser.add_argument("--psd_test_weight_type1", type=str, default="weight_psd/fine_tuned_type_1",
                    help="The path of the trained weights of fine-tuned to perpendicular type.")

parser.add_argument("--psd_test_weight_type2", type=str, default="weight_psd/fine_tuned_type_2",
                    help="The path of the trained weights of fine-tuned to diagonal type.")

parser.add_argument("--threshold_score", type=float, default=0.8,
                    help="Threshold of prediction which is determined TRUE")

args = parser.parse_args()

pcr_train.evaluate(os.path.join(args.data_path, "test"), args.pcr_test_weight)

psd_test.evaluate(args.psd_test_weight_type0, "result/result_pcr_type_0.txt", "result/result_psd_type_0.txt")
psd_test.evaluate(args.psd_test_weight_type1, "result/result_pcr_type_1.txt", "result/result_psd_type_1.txt")
psd_test.evaluate(args.psd_test_weight_type2, "result/result_pcr_type_2.txt", "result/result_psd_type_2.txt")

result_files = ["result/result_psd_type_0.txt","result/result_psd_type_1.txt","result/result_psd_type_2.txt"]
merge_result.merge_result(os.path.join(args.data_path, "test.txt"), result_files, "result/result.txt")
eval_utils.eval_result_file(os.path.join(args.data_path, "test.txt"), "result/result.txt", args.threshold_score)

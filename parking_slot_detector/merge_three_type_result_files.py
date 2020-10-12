import os

from parking_slot_detector.utils.eval_utils import parse_line, parse_result_line, parse_gt_quadrangle


def merge_result(eval_file, result_files, merged_file):
    gt_dict = parse_gt_quadrangle(eval_file, [256, 768])
    file_dict = {}
    f = open(eval_file)
    for line in f:
        img_id, pic_path, _, _, _, _, _, _ = parse_line(line)
        file_dict[img_id] = pic_path
    f.close()
    result_dict = {}

    for result_file in result_files:
        f = open(result_file)
        lines = f.readlines()
        for line in lines:
            old_img_id, pic_path, boxes, labels, confidences, quads = parse_result_line(line)
            filename = os.path.basename(pic_path)
            result_dict[filename] = line[len(str(old_img_id)):]
        f.close()

    merged_f = open(merged_file, 'wt')

    for id, objects in gt_dict.items():
        pic_path = file_dict[id]
        filename = os.path.basename(pic_path)
        if filename in result_dict:
            new_line = str(id) + result_dict[filename]
            merged_f.write(new_line)
        else:
            merged_f.write(f'{id} {pic_path}\n')

    merged_f.close()

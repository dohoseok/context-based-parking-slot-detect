import os
import cv2

from dataset.dataset_utils import get_data_from_our_txt


def create_detector_data_file(data_path, target_file, include_non_park=False):
    f = open(target_file, 'wt')
    image_path = os.path.join(data_path, "image")
    label_path = os.path.join(data_path, "label")

    files = os.listdir(image_path)
    file_idx = 0
    for file in files:
        jpg_file = os.path.join(image_path, file)
        img = cv2.imread(jpg_file)
        height, width, channel = img.shape
        label_file = os.path.join(label_path, file.replace(".jpg", ".txt"))
        if not os.path.exists(label_file):
            continue

        type, angle, box_list = get_data_from_our_txt(label_file, split=' ')

        if angle < - 180 :
            angle += 360
        if angle > 180:
            angle -= 360

        if include_non_park == False:
            if type == 3 or len(box_list) == 0:
                continue

        file_path = image_path + "/" + file
        file_str = f'{file_idx} {file_path} {width} {height} {angle} '
        f.write(file_str)

        for box in box_list:
            box_str = '{0} {1} {2} {3} {4} {5} {6} {7} {8} '.format(*box)
            f.write(box_str)

        file_idx += 1
        if file != files[-1]:
            f.write('\n')
    f.close()


def create_detector_data_file_fine_tune(src_path, dst_path):
    dst_paths =[]
    for i in range(4):
        dst_path = os.path.join(dst_path, f't_{i}')
        os.makedirs(dst_path, exist_ok=True)
        dst_paths.append(dst_path)

    for sub in ["test", "train"]:
        print("{} start".format(sub))
        main_f = open(os.path.join(src_path, "{}.txt".format(sub)))
        label_path = os.path.join(src_path, sub, "label")

        f0 = open(os.path.join(dst_paths[0], sub+".txt"), 'wt')
        f1 = open(os.path.join(dst_paths[1], sub+".txt"), 'wt')
        f2 = open(os.path.join(dst_paths[2], sub+".txt"), 'wt')
        f3 = open(os.path.join(dst_paths[3], sub+".txt"), 'wt')

        f_list = [f0, f1, f2, f3]
        type_count = [0] * 4

        lines = main_f.readlines()

        for idx, line in enumerate(lines):
            if idx % 1000 == 0:
                print("idx= {}".format(idx))
            s = line.strip().split(' ')
            jpg_file = s[1]
            filename = os.path.basename(jpg_file)[:-3] + "txt"
            label_file = os.path.join(label_path, filename)
            type, _, _ = get_data_from_our_txt(label_file)
            s[0] = str(type_count[type])
            type_count[type] += 1
            for temp in s:
                f_list[type].write(temp)
                f_list[type].write(' ')
            f_list[type].write('\n')

        for f in f_list:
            f.close()
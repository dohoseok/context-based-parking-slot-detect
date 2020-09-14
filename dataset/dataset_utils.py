import os
import cv2


def get_data_from_our_txt(file, format=None, split = '\t'):
    f = open(file)
    lines = f.readlines()

    type = int(lines[0][0])
    angle = int(lines[1][:-1])
    box_list = []

    for line in lines[2:]:
        box_data = line.strip().split(split)
        if format != None:
            for idx, box in enumerate(box_data):
                box_data[idx] = format(box)
        box_list.append(box_data)
    f.close()

    return type, angle, box_list


def write_data_to_our_txt(file, type, angle, box_list):
    f = open(file, 'wt')
    f.write(str(type) + '\n')
    f.write(str(int(round(angle, 0))) + '\n')
    for box in box_list:
        for coord in box:
            f.write(str(int(round(coord, 0))))
            f.write('\t')
        f.write('\n')
    f.close()
    return


def create_detector_data_file(data_path, target_file, include_non_park=False):
    image_path = os.path.join(data_path, "image")
    label_path = os.path.join(data_path, "label")

    f = open(target_file, 'wt')

    files = os.listdir(image_path)
    file_idx = 0
    for file in files:
        jpg_file = os.path.join(image_path, file)
        img = cv2.imread(jpg_file)
        height, width, channel = img.shape
        label_file = os.path.join(label_path, file.replace(".jpg", ".txt"))
        if not os.path.exists(label_file):
            continue

        type, angle, box_list = get_data_from_our_txt(label_file, split='\t')

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
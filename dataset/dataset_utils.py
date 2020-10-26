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


def cut_border_points(box_data_list, max_w=239, max_h=719):
    for idx, box_data in enumerate(box_data_list):
        parked, x1, y1, x2, y2, x3, y3, x4, y4 = box_data
        if x4 == x1:
            sep_1_ratio = 0
        else:
            sep_1_ratio = (y4 - y1) / (x4 - x1)
        if y4 == y1:
            sep_1_inv = 0
        else:
            sep_1_inv = (x4 - x1) / (y4 - y1)
        if x2 == x3:
            sep_2_ratio = 0
        else:
            sep_2_ratio = (y3 - y2) / (x3 - x2)
        if y2 == y3:
            sep_2_inv = 0
        else:
            sep_2_inv = (x3 - x2) / (y3 - y2)

        # while(min([x1,x2,x3,x4,y1,y2,y3,y4])<0 or max([x1,x2,x3,x4])>max_w or max([y1,y2,y3,y4])>max_h):
        #     print(x1,x2,x3,x4,y1,y2,y3,y4, sep_1_ratio, sep_1_inv, sep_2_ratio, sep_2_inv)
        if x1 < 0:
            y1 -= sep_1_ratio * x1
            x1 = 0
        elif x1 > max_w:
            y1 += sep_1_ratio * (max_w - x1)
            x1 = max_w

        if x2 < 0:
            y2 -= sep_2_ratio * x2
            x2 = 0
        elif x2 > max_w:
            y2 += sep_2_ratio * (max_w - x2)
            x2 = max_w

        if x3 < 0:
            y3 -= sep_2_ratio * x3
            x3 = 0
        elif x3 > max_w:
            y3 += sep_2_ratio * (max_w - x3)
            x3 = max_w

        if x4 < 0:
            y4 -= sep_1_ratio * x4
            x4 = 0
        elif x4 > max_w:
            y4 += sep_1_ratio * (max_w - x4)
            x4 = max_w

        if y1 < 0:
            x1 -= sep_1_inv * y1
            y1 = 0
        elif y1 > max_h:
            y1 += sep_1_inv * (max_h - x1)
            x1 = max_h

        if y2 < 0:
            x2 -= sep_2_inv * y2
            y2 = 0
        elif y2 > max_h:
            x2 += sep_2_inv * (max_h - y2)
            y2 = max_h

        if y3 < 0:
            x3 -= sep_2_inv * y3
            y3 = 0
        elif y3 > max_h:
            x3 += sep_2_inv * (max_h - y3)
            y3 = max_h

        if y4 < 0:
            x4 -= sep_1_inv * y4
            y4 = 0
        elif y4 > max_h:
            x4 += sep_1_inv * (max_h - y4)
            y4 = max_h
        box_data_list[idx] = [parked, round(x1), round(y1), round(x2), round(y2), round(x3), round(y3), round(x4),
                              round(y4)]
    return box_data_list


def write_data_to_our_txt(file, type, angle, box_list):
    f = open(file, 'wt')
    f.write(str(type) + '\n')
    f.write(str(int(round(angle, 0))) + '\n')
    box_list = cut_border_points(box_list, max_w=255, max_h=767)
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
        if file_idx % 1000 == 0:
            print("file_idx = ", file_idx)
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


def create_detector_data_file_fine_tune(data_path):
    dst_paths =[]
    for i in range(4):
        dst_path = os.path.join(data_path, f't{i}')
        os.makedirs(dst_path, exist_ok=True)
        dst_paths.append(dst_path)

    for sub in ["test", "train"]:
        print("{} start".format(sub))
        main_f = open(os.path.join(data_path, "{}.txt".format(sub)))
        label_path = os.path.join(data_path, sub, "label")

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

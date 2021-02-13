import cv2
import os
import numpy as np

from dataset.dataset_utils import *

_WIDTH = 256
_HEIGHT = 768


def rotate_box(bb, cx, cy, h, w, rot_angle):
    # new_bb = list(bb)
    new_bb = []
    for i,coord in enumerate(bb):
        # opencv calculates standard transformation matrix
        M = cv2.getRotationMatrix2D((cx, cy), rot_angle, 1.0)
        # compute the new bounding dimensions of the image
        M[0, 2] += (w / 2) - cx
        M[1, 2] += (h / 2) - cy
        # Prepare the vector to be transformed
        v = [coord[0],coord[1],1]
        # Perform the actual rotation and return the image
        calculated = np.dot(M,v)
        new_bb.append(int(calculated[0]))
        new_bb.append(int(calculated[1]))
    return new_bb


def data_augment(jpg_file, txt_file, dst_jpg, dst_txt, rot_angle, flip=False):
    image = cv2.imread(jpg_file)
    height, width, channel = image.shape

    if flip:
        image = cv2.flip(image, 0)

    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), -rot_angle, 1)
    image_rot = cv2.warpAffine(image, matrix, (width, height), borderMode= cv2.BORDER_CONSTANT, borderValue = [128, 128, 128])

    type, angle, box_list = get_data_from_our_txt(txt_file, int, '\t')
    if type == 3:
        new_angle = 0
    else:
        if flip:
            new_angle = -angle + rot_angle
        else:
            new_angle = angle + rot_angle
    if new_angle < -180:
        new_angle += 360
    elif new_angle > 180:
        new_angle -= 360
    new_box_list = []

    for p in box_list:
        if flip:
            bb = [[int(p[3]), height -1 - int(p[4])], [int(p[1]), height - 1 - int(p[2])], [int(p[7]), height-1 - int(p[8])], [int(p[5]), height-1 - int(p[6])]]
        else:
            bb = [[int(p[1]), int(p[2])], [int(p[3]), int(p[4])], [int(p[5]), int(p[6])], [int(p[7]), int(p[8])]]
        new_bb = rotate_box(bb, _WIDTH/2, _HEIGHT/2, _HEIGHT, _WIDTH, -rot_angle)
        new_bb.insert(0, int(p[0]))
        if flip:
            new_box_list.insert(0, new_bb)
        else:
            new_box_list.append(new_bb)

    write_data_to_our_txt(dst_txt, type, new_angle, new_box_list)
    cv2.imwrite(dst_jpg, image_rot)


def run(data_path):
    image_path = os.path.join(data_path, "image")
    label_path = os.path.join(data_path, "label")
    jpg_files = os.listdir(image_path)
    txt_files = os.listdir(label_path)

    for jpg_file, txt_file in zip(jpg_files, txt_files):
        src_image_file = os.path.join(image_path, jpg_file)
        src_label_file = os.path.join(label_path, txt_file)
        print(jpg_file)
        for angle in range(-5, 6):
            for flip in [True, False]:
                if angle == 0 and flip == False:
                    continue
                if flip:
                    dst_jpg_file = jpg_file.replace(".jpg", "_flip_%d.jpg"%angle)
                    dst_txt_file = txt_file.replace(".txt", "_flip_%d.txt"%angle)
                else:
                    dst_jpg_file = jpg_file.replace(".jpg", "_%d.jpg" % angle)
                    dst_txt_file = txt_file.replace(".txt", "_%d.txt" % angle)
                dst_image_file = os.path.join(image_path, dst_jpg_file)
                dst_label_file = os.path.join(label_path, dst_txt_file)
                data_augment(src_image_file, src_label_file, dst_image_file, dst_label_file, angle, flip)

# coding: utf-8

from __future__ import division, print_function

import numpy as np
import cv2
import sys
import random
import math

DATA_LEN = 14


def parse_line(line):
    if 'str' not in str(type(line)):
        line = line.decode()
    s = line.strip().split(' ')
    # assert len(s) > 8, 'Annotation error! Please check your annotation file. Make sure there is at least one target object in each image.'
    line_idx = int(s[0])
    pic_path = s[1]
    img_width = int(s[2])
    img_height = int(s[3])
    img_angle = int(s[4])
    s = s[5:]
    assert len(s) % 9 == 0, 'Annotation error! Please check your annotation file. Maybe partially missing some coordinates?'
    box_cnt = len(s) // 9
    boxes = []
    labels = []
    quads = []
    for i in range(box_cnt):
        label, x1, y1, x2, y2, x3, y3, x4, y4 = int(s[i * 9]), float(s[i * 9 + 1]), float(s[i * 9 + 2]), float(
            s[i * 9 + 3]), float(s[i * 9 + 4]), float(s[i * 9 + 5]), float(s[i * 9 + 6]), float(
            s[i * 9 + 7]), float(s[i * 9 + 8])
        x_min, y_min, x_max, y_max = min(x1, x2, x3, x4), min(y1, y2, y3, y4), max(x1, x2, x3, x4), max(y1, y2, y3, y4)
        boxes.append([x_min, y_min, x_max, y_max])
        labels.append(label)
        if label !=0 and label != 1:
            print("Error labe=", label, line)
        quads.append([x1, y1, x2, y2, x3, y3, x4, y4])
    boxes = np.asarray(boxes, np.float32)
    labels = np.asarray(labels, np.int64)
    quads = np.asarray(quads, np.float32)
    return line_idx, pic_path, boxes, labels, img_width, img_height, quads, img_angle


def parse_result_line(line):
    if 'str' not in str(type(line)):
        line = line.decode()
    s = line.strip().split(' ')
    line_idx = int(s[0])
    pic_path = s[1]
    s = s[2:]
    boxes = []
    labels = []
    quads = []
    confidences = []
    if len(s) < 10:
        return line_idx, pic_path, boxes, labels, confidences, quads
    assert len(s) >= 10, 'Annotation error! Please check your annotation file. Make sure there is at least one target object in each image.'

    assert len(s) % 10 == 0, 'Annotation error! Please check your annotation file. Maybe partially missing some coordinates?'
    box_cnt = len(s) // 10
    for i in range(box_cnt):
        label, confidence, x1, y1, x2, y2, x3, y3, x4, y4 = int(s[i * 10]), float(s[i * 10 + 1]), float(s[i * 10 + 2]), float(
            s[i * 10 + 3]), float(s[i * 10 + 4]), float(s[i * 10 + 5]), float(s[i * 10 + 6]), float(
            s[i * 10 + 7]), float(s[i * 10 + 8]), float(s[i * 10 + 9])
        x_min, y_min, x_max, y_max = min(x1, x2, x3, x4), min(y1, y2, y3, y4), max(x1, x2, x3, x4), max(y1, y2, y3, y4)
        boxes.append([x_min, y_min, x_max, y_max])
        labels.append(label)
        confidences.append(confidence)
        quads.append([x1, y1, x2, y2, x3, y3, x4, y4])
    boxes = np.asarray(boxes, np.float32)
    labels = np.asarray(labels, np.int64)
    confidences = np.asarray(confidences, np.float32)
    quads = np.asarray(quads, np.float32)
    return line_idx, pic_path, boxes, labels, confidences, quads


def process_box(boxes, labels, img_size, class_num, anchors, quads, angle=0):
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    y_true_13 = np.zeros((img_size[1] // 32, img_size[0] // 32, 3, DATA_LEN + class_num), np.float32)
    y_true_26 = np.zeros((img_size[1] // 16, img_size[0] // 16, 3, DATA_LEN + class_num), np.float32)
    y_true_52 = np.zeros((img_size[1] // 8, img_size[0] // 8, 3, DATA_LEN + class_num), np.float32)

    # mix up weight default to 1.
    y_true_13[..., -1] = 1.
    y_true_26[..., -1] = 1.
    y_true_52[..., -1] = 1.

    y_true = [y_true_13, y_true_26, y_true_52]

    if len(quads) == 0:
        return y_true

    box_centers = (boxes[:, 0:2] + boxes[:, 2:4]) / 2

    rot_mat = [[math.cos(angle * math.pi / 180), -math.sin(angle * math.pi / 180)],
               [math.sin(angle * math.pi / 180), math.cos(angle * math.pi / 180)]]

    for idx, quad in enumerate(quads):
        rot_p1 = np.matmul(rot_mat, quad[0:2])
        rot_p2 = np.matmul(rot_mat, quad[2:4])
        rot_p3 = np.matmul(rot_mat, quad[4:6])
        rot_p4 = np.matmul(rot_mat, quad[6:8])

        boxes[idx] = [min(rot_p1[0], rot_p2[0], rot_p3[0], rot_p4[0]), min(rot_p1[1], rot_p2[1], rot_p3[1], rot_p4[1]),
                      max(rot_p1[0], rot_p2[0], rot_p3[0], rot_p4[0]), max(rot_p1[1], rot_p2[1], rot_p3[1], rot_p4[1]), boxes[idx][-1]]

    box_sizes = boxes[:, 2:4] - boxes[:, 0:2]
    box_sizes = np.expand_dims(box_sizes, 1)
    quads = np.expand_dims(quads, 1)

    mins = np.maximum(- box_sizes / 2, - anchors / 2)
    maxs = np.minimum(box_sizes / 2, anchors / 2)
    whs = maxs - mins
    iou = (whs[:, :, 0] * whs[:, :, 1]) / (
                box_sizes[:, :, 0] * box_sizes[:, :, 1] + anchors[:, 0] * anchors[:, 1] - whs[:, :, 0] * whs[:, :,
                                                                                                         1] + 1e-10)
    best_match_idx = np.argmax(iou, axis=1)

    ratio_dict = {1.: 8., 2.: 16., 3.: 32.}
    for i, idx in enumerate(best_match_idx):
        # idx: 0,1,2 ==> 2; 3,4,5 ==> 1; 6,7,8 ==> 0
        feature_map_group = 2 - idx // 3
        # scale ratio: 0,1,2 ==> 8; 3,4,5 ==> 16; 6,7,8 ==> 32
        ratio = ratio_dict[np.ceil((idx + 1) / 3.)]
        x = int(np.floor(box_centers[i, 0] / ratio))
        y = int(np.floor(box_centers[i, 1] / ratio))
        k = anchors_mask[feature_map_group].index(idx)
        c = labels[i]

        y_true[feature_map_group][y, x, k, :2] = box_centers[i]
        y_true[feature_map_group][y, x, k, 2:4] = box_sizes[i]
        y_true[feature_map_group][y, x, k, 4] = 1.
        y_true[feature_map_group][y, x, k, 5 + c] = 1.
        y_true[feature_map_group][y, x, k, 5 + class_num: 5 + class_num + 8] = quads[i]
        y_true[feature_map_group][y, x, k, -1] = boxes[i, -1]

    return y_true_13, y_true_26, y_true_52


def parse_data(line, class_num, img_size, anchors):
    img_idx, pic_path, boxes, labels, _, _, quads, angle = parse_line(line)
    img = cv2.imread(pic_path)
    if len(quads) >0:
        boxes = np.concatenate((boxes, np.full(shape=(boxes.shape[0], 1), fill_value=1., dtype=np.float32)), axis=-1)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

    # the input of yolo_v3 should be in range 0~1
    img = img / 255.

    y_true_13, y_true_26, y_true_52 = process_box(boxes, labels, img_size, class_num, anchors, quads, angle)

    return img_idx, img, y_true_13, y_true_26, y_true_52, angle


def get_batch_data(batch_line, class_num, img_size, anchors, mix_up = False):
    img_idx_batch, img_batch, y_true_13_batch, y_true_26_batch, y_true_52_batch, img_angle_batch = [], [], [], [], [], []

    for line in batch_line:
        img_idx, img, y_true_13, y_true_26, y_true_52, img_angle = parse_data(line, class_num, img_size, anchors)

        img_idx_batch.append(img_idx)
        img_batch.append(img)
        y_true_13_batch.append(y_true_13)
        y_true_26_batch.append(y_true_26)
        y_true_52_batch.append(y_true_52)
        img_angle_batch.append(img_angle)

    img_idx_batch, img_batch, y_true_13_batch, y_true_26_batch, y_true_52_batc, img_angle_batch = np.asarray(img_idx_batch, np.int64), np.asarray(img_batch), np.asarray(y_true_13_batch), np.asarray(y_true_26_batch), np.asarray(y_true_52_batch), np.asarray(img_angle_batch, np.int64)

    return img_idx_batch, img_batch, y_true_13_batch, y_true_26_batch, y_true_52_batch, img_angle_batch

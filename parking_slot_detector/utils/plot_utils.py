# coding: utf-8

from __future__ import division, print_function

import cv2
import random
import numpy as np


def get_color_table(class_num, seed=100):
    random.seed(seed)
    color_table = {}
    for i in range(class_num):
        color_table[i] = [random.randint(0, 255) for _ in range(3)]
    return color_table


def plot_one_box(img, coord, label=None, color=None, line_thickness=None):
    '''
    coord: [x_min, y_min, x_max, y_max] format coordinates.
    img: img to plot on.
    label: str. The label name.
    color: int. color index.
    line_thickness: int. rectangle line thickness.
    '''
    tl = line_thickness or int(round(0.002 * max(img.shape[0:2])))  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=float(tl) / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, float(tl) / 3, [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA)

def plot_one_quad(img, quad, label=None, color=None, line_thickness=None):
    '''
    coord: [x_min, y_min, x_max, y_max] format coordinates.
    img: img to plot on.
    label: str. The label name.
    color: int. color index.
    line_thickness: int. rectangle line thickness.
    '''
    tl = line_thickness or int(round(0.002 * max(img.shape[0:2])))  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2, c3, c4 = (int(quad[0]), int(quad[1])), (int(quad[2]), int(quad[3])), (int(quad[4]), int(quad[5])), (int(quad[6]), int(quad[7]))
    pts = np.array([quad[0:2], quad[2:4], quad[4:6], quad[6:8], quad[0:2]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(img, [pts], False, color, thickness=tl)
    if label:
        tf = max(tl - 1, 2)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=float(tl) / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        # cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label, ((c1[0]+c3[0]-t_size[0])//2, (c1[1]+c3[1]+t_size[1])//2), 0, float(tl) / 3, color, thickness=tf, lineType=cv2.LINE_AA)


def plot_one_quad_center(img, quad, center, label=None, color=None, line_thickness=None):
    '''
    coord: [x_min, y_min, x_max, y_max] format coordinates.
    img: img to plot on.
    label: str. The label name.
    color: int. color index.
    line_thickness: int. rectangle line thickness.
    '''
    tl = line_thickness or int(round(0.002 * max(img.shape[0:2])))  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2, c3, c4 = (int(quad[0]), int(quad[1])), (int(quad[2]), int(quad[3])), (int(quad[4]), int(quad[5])), (int(quad[6]), int(quad[7]))
    pts = np.array([quad[0:2], quad[2:4], quad[4:6], quad[6:8], quad[0:2]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(img, [pts], False, color, thickness=tl)
    # print(center)
    # center_pt = cv2.Point(center[0], center[1])
    # center_pt = center_pt.reshape((1, 2))
    # print(center_pt)

    if label:
        tf = max(tl - 1, 2)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=float(tl) / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        # cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label, ((c1[0]+c3[0]-t_size[0])//2, (c1[1]+c3[1]+t_size[1])//2), 0, float(tl) / 3, color, thickness=tf, lineType=cv2.LINE_AA)
    center_color = [255 - color[0], 255 - color[1], 255 - color[2]]
    cv2.circle(img, (center[0], center[1]), 5, center_color, -1)
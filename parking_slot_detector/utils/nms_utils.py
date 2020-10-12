# coding: utf-8

from __future__ import division, print_function

import numpy as np
import tensorflow as tf
import math

def rot_nms(filter_score, filter_quads, max_boxes, nms_thresh):
    max_score_idx = tf.argmax(filter_score)
    best_quad = filter_quads[max_score_idx]
    y_diff = best_quad[..., 7] + best_quad[..., 5] - best_quad[..., 3] - best_quad[..., 1]
    x_diff = best_quad[..., 6] + best_quad[..., 4] - best_quad[..., 2] - best_quad[..., 0]
    angle = tf.atan2(y_diff, x_diff)
    temp_quads = tf.reshape(filter_quads, [-1, 4, 2])

    rot_x = tf.stack([tf.cos(angle), -tf.sin(angle)], -1)
    rot_y = tf.stack([tf.sin(angle), tf.cos(angle)], -1)
    rot_mat = tf.stack([rot_x, rot_y], -2)
    # rot_mat_repeat = tf.stack([rot_mat, rot_mat, rot_mat, rot_mat], -2)
    rot_quads = tf.einsum('jk,lij->lik', rot_mat, temp_quads)
    rot_quads = tf.reshape(rot_quads, [-1, 8])
    rot_boxes = tf.stack(
        [tf.minimum(tf.minimum(rot_quads[..., 0], rot_quads[..., 2]), tf.minimum(rot_quads[..., 4], rot_quads[..., 6])),
         tf.minimum(tf.minimum(rot_quads[..., 1], rot_quads[..., 3]), tf.minimum(rot_quads[..., 5], rot_quads[..., 7])),
         tf.maximum(tf.maximum(rot_quads[..., 0], rot_quads[..., 2]), tf.maximum(rot_quads[..., 4], rot_quads[..., 6])),
         tf.maximum(tf.maximum(rot_quads[..., 1], rot_quads[..., 3]),
                    tf.maximum(rot_quads[..., 5], rot_quads[..., 7]))],
        axis=-1)

    nms_indices = tf.image.non_max_suppression(boxes=rot_boxes,
                                               scores=filter_score,
                                               max_output_size=max_boxes,
                                               iou_threshold=nms_thresh, name='nms_indices')
    return nms_indices

def gpu_nms(quads, scores, num_classes, max_boxes=50, score_thresh=0.5, nms_thresh=0.5, apply_rotate=True):
    """
    Perform NMS on GPU using TensorFlow.

    params:
        boxes: tensor of shape [1, 10647, 4] # 10647=(13*13+26*26+52*52)*3, for input 416*416 image
        scores: tensor of shape [1, 10647, num_classes], score=conf*prob
        num_classes: total number of classes
        max_boxes: integer, maximum number of predicted boxes you'd like, default is 50
        score_thresh: if [ highest class probability score < score_threshold]
                        then get rid of the corresponding box
        nms_thresh: real value, "intersection over union" threshold used for NMS filtering
    """

    boxes_list, label_list, score_list, quads_list = [], [], [], []
    max_boxes = tf.constant(max_boxes, dtype='int32')

    quads = tf.reshape(quads, [-1, 8])
    # boxes = tf.stack([tf.minimum(quads[..., 0],quads[..., 2]), tf.minimum(quads[..., 1],quads[..., 7]), tf.maximum(quads[..., 4],quads[..., 6]), tf.maximum(quads[..., 3],quads[..., 5])], axis=-1)
    boxes = tf.stack([tf.minimum(tf.minimum(quads[..., 0],quads[..., 2]), tf.minimum(quads[..., 4],quads[..., 6])),
                      tf.minimum(tf.minimum(quads[..., 1],quads[..., 3]), tf.minimum(quads[..., 5],quads[..., 7])),
                      tf.maximum(tf.maximum(quads[..., 0],quads[..., 2]), tf.maximum(quads[..., 4],quads[..., 6])),
                      tf.maximum(tf.maximum(quads[..., 1],quads[..., 3]), tf.maximum(quads[..., 5],quads[..., 7]))],
                     axis=-1)
    # boxes = tf.gather(quads, indices = [0, 1, 4, 5], axis=-1)
    # since we do nms for single image, then reshape it
    boxes = tf.reshape(boxes, [-1, 4]) # '-1' means we don't konw the exact number of boxes
    score = tf.reshape(scores, [-1, num_classes])
    labels = tf.argmax(score, axis=1)
    score = tf.reduce_max(score, axis=1)
    # score = tf.reduce_max(score[:,0], score[:,1])
    #
    # print("boxes size", tf.size(boxes))
    # print("quads size", tf.size(quads))

    # Step 1: Create a filtering mask based on "box_class_scores" by using "threshold".
    mask = tf.greater_equal(score, tf.constant(score_thresh))
    # Step 2: Do non_max_suppression for each class
    # for i in range(num_classes):
        # Step 3: Apply the mask to scores, boxes and pick them out
    filter_labels = tf.boolean_mask(labels, mask)
    filter_boxes = tf.boolean_mask(boxes, mask)
    filter_score = tf.boolean_mask(score, mask)
    filter_quads = tf.boolean_mask(quads, mask)

    if apply_rotate:
        nms_indices = tf.cond(tf.greater(tf.shape(filter_score)[0], 0),
                              lambda:rot_nms(filter_score, filter_quads, max_boxes, nms_thresh),
                              lambda:tf.image.non_max_suppression(boxes=filter_boxes,
                                                       scores=filter_score,
                                                       max_output_size=max_boxes,
                                                       iou_threshold=nms_thresh, name='nms_indices')
                              )
    else:
        nms_indices = tf.image.non_max_suppression(boxes=filter_boxes,
                                                   scores=filter_score,
                                                   max_output_size=max_boxes,
                                                   iou_threshold=nms_thresh, name='nms_indices')



    label_list.append(tf.gather(filter_labels, nms_indices))
    boxes_list.append(tf.gather(filter_boxes, nms_indices))
    score_list.append(tf.gather(filter_score, nms_indices))
    quads_list.append(tf.gather(filter_quads, nms_indices))

    boxes = tf.concat(boxes_list, axis=0)
    score = tf.concat(score_list, axis=0)
    label = tf.concat(label_list, axis=0)
    quad = tf.concat(quads_list, axis=0)

    return boxes, score, label, quad


def py_nms(boxes, scores, max_boxes=50, iou_thresh=0.5):
    """
    Pure Python NMS baseline.

    Arguments: boxes: shape of [-1, 4], the value of '-1' means that dont know the
                      exact number of boxes
               scores: shape of [-1,]
               max_boxes: representing the maximum of boxes to be selected by non_max_suppression
               iou_thresh: representing iou_threshold for deciding to keep boxes
    """
    assert boxes.shape[1] == 4 and len(scores.shape) == 1

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_thresh)[0]
        order = order[inds + 1]

    return keep[:max_boxes]


def cpu_nms(boxes, scores, num_classes, max_boxes=50, score_thresh=0.5, iou_thresh=0.5):
    """
    Perform NMS on CPU.
    Arguments:
        boxes: shape [1, 10647, 4]
        scores: shape [1, 10647, num_classes]
    """

    boxes = boxes.reshape(-1, 4)
    scores = scores.reshape(-1, num_classes)
    # Picked bounding boxes
    picked_boxes, picked_score, picked_label = [], [], []

    for i in range(num_classes):
        indices = np.where(scores[:,i] >= score_thresh)
        filter_boxes = boxes[indices]
        filter_scores = scores[:,i][indices]
        if len(filter_boxes) == 0: 
            continue
        # do non_max_suppression on the cpu
        indices = py_nms(filter_boxes, filter_scores,
                         max_boxes=max_boxes, iou_thresh=iou_thresh)
        picked_boxes.append(filter_boxes[indices])
        picked_score.append(filter_scores[indices])
        picked_label.append(np.ones(len(indices), dtype='int32')*i)
    if len(picked_boxes) == 0: 
        return None, None, None

    boxes = np.concatenate(picked_boxes, axis=0)
    score = np.concatenate(picked_score, axis=0)
    label = np.concatenate(picked_label, axis=0)

    return boxes, score, label
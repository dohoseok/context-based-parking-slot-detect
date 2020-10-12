# coding: utf-8

from __future__ import division, print_function

import numpy as np
from collections import Counter
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from matplotlib import pyplot

from parking_slot_detector.utils.nms_utils import cpu_nms
from parking_slot_detector.utils.data_utils import parse_line, parse_result_line


def calc_iou(pred_boxes, true_boxes):
    pred_boxes = np.expand_dims(pred_boxes, -2)
    true_boxes = np.expand_dims(true_boxes, 0)

    intersect_mins = np.maximum(pred_boxes[..., :2], true_boxes[..., :2])
    intersect_maxs = np.minimum(pred_boxes[..., 2:], true_boxes[..., 2:])
    intersect_wh = np.maximum(intersect_maxs - intersect_mins, 0.)

    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    pred_box_wh = pred_boxes[..., 2:] - pred_boxes[..., :2]
    pred_box_area = pred_box_wh[..., 0] * pred_box_wh[..., 1]
    true_boxes_wh = true_boxes[..., 2:] - true_boxes[..., :2]
    true_boxes_area = true_boxes_wh[..., 0] * true_boxes_wh[..., 1]

    iou = intersect_area / (pred_box_area + true_boxes_area - intersect_area + 1e-10)

    return iou


def evaluate_on_gpu(sess, gpu_nms_op, pred_quads_flag, pred_scores_flag, y_pred, y_true, num_classes, iou_thresh=0.5, calc_now=True):
    '''
    Given y_pred and y_true of a batch of data, get the recall and precision of the current batch.
    This function will perform gpu operation on the GPU.
    '''

    num_images = y_true[0].shape[0]
    true_labels_dict = {i: 0 for i in range(num_classes)}  # {class: count}
    pred_labels_dict = {i: 0 for i in range(num_classes)}
    true_positive_dict = {i: 0 for i in range(num_classes)}

    for i in range(num_images):
        true_labels_list, true_boxes_list = [], []
        for j in range(3):  # three feature maps
            # shape: [13, 13, 3, 80]
            true_probs_temp = y_true[j][i][..., 5:5+num_classes]
            # shape: [13, 13, 3, 4] (x_center, y_center, w, h)
            true_boxes_temp = y_true[j][i][..., 0:4]

            # [13, 13, 3]
            object_mask = true_probs_temp.sum(axis=-1) > 0

            # [V, 80] V: Ground truth number of the current image
            true_probs_temp = true_probs_temp[object_mask]
            # [V, 4]
            true_boxes_temp = true_boxes_temp[object_mask]

            # [V], labels, each from 0 to 79
            true_labels_list += np.argmax(true_probs_temp, axis=-1).tolist()
            # [V, 4] (x_center, y_center, w, h)
            true_boxes_list += true_boxes_temp.tolist()

        if len(true_labels_list) != 0:
            for cls, count in Counter(true_labels_list).items():
                true_labels_dict[cls] += count

        # [V, 4] (xmin, ymin, xmax, ymax)
            true_boxes = np.array(true_boxes_list)
            box_centers, box_sizes = true_boxes[:, 0:2], true_boxes[:, 2:4]
            true_boxes[:, 0:2] = box_centers - box_sizes / 2.
            true_boxes[:, 2:4] = true_boxes[:, 0:2] + box_sizes
        else:
            continue

        # [1, xxx, 4]
        # pred_boxes = y_pred[0][i:i + 1]
        pred_confs = y_pred[0][i:i + 1]
        pred_probs = y_pred[1][i:i + 1]
        pred_quads = y_pred[2][i:i + 1]

        # pred_boxes: [N, 4]
        # pred_confs: [N]
        # pred_labels: [N]
        # N: Detected box number of the current image
        pred_boxes, pred_confs, pred_labels, pred_quads = sess.run(gpu_nms_op,
                                                       feed_dict={pred_quads_flag: pred_quads,
                                                                  pred_scores_flag: pred_confs * pred_probs})
        # len: N
        pred_labels_list = [] if pred_labels is None else pred_labels.tolist()
        if pred_labels_list == []:
            continue

        # calc iou
        # [N, V]
        iou_matrix = calc_iou(pred_boxes, true_boxes)
        # [N]
        max_iou_idx = np.argmax(iou_matrix, axis=-1)

        correct_idx = []
        correct_conf = []
        for k in range(max_iou_idx.shape[0]):
            pred_labels_dict[pred_labels_list[k]] += 1
            match_idx = max_iou_idx[k]  # V level
            if iou_matrix[k, match_idx] > iou_thresh and true_labels_list[match_idx] == pred_labels_list[k]:
                if match_idx not in correct_idx:
                    correct_idx.append(match_idx)
                    correct_conf.append(pred_confs[k])
                else:
                    same_idx = correct_idx.index(match_idx)
                    if pred_confs[k] > correct_conf[same_idx]:
                        correct_idx.pop(same_idx)
                        correct_conf.pop(same_idx)
                        correct_idx.append(match_idx)
                        correct_conf.append(pred_confs[k])

        for t in correct_idx:
            true_positive_dict[true_labels_list[t]] += 1

    if calc_now:
        # avoid divided by 0
        recall = sum(true_positive_dict.values()) / (sum(true_labels_dict.values()) + 1e-6)
        precision = sum(true_positive_dict.values()) / (sum(pred_labels_dict.values()) + 1e-6)

        return recall, precision
    else:
        return true_positive_dict, true_labels_dict, pred_labels_dict


def get_preds_gpu(sess, gpu_nms_op, pred_quads_flag, pred_scores_flag, image_ids, y_pred):
    '''
    Given the y_pred of an input image, get the predicted bbox and label info.
    return:
        pred_content: 2d list.
    '''
    image_id = image_ids[0]

    # keep the first dimension 1
    # pred_boxes = y_pred[0][0:1]
    pred_confs = y_pred[0][0:1]
    pred_probs = y_pred[1][0:1]
    pred_quads = y_pred[2][0:1]

    # print(pred_confs, pred_probs, pred_quads)
    # print(len(pred_quads))

    boxes, scores, labels, quads = sess.run(gpu_nms_op,
                                     feed_dict={pred_quads_flag: pred_quads,
                                                pred_scores_flag: pred_confs * pred_probs})

    # print(len(boxes))
    # print(boxes)
    # print(quads)
    pred_content = []
    for i in range(len(labels)):
        x_min, y_min, x_max, y_max = boxes[i]
        score = scores[i]
        label = labels[i]
        quad = quads[i]
        pred_content.append([image_id, x_min, y_min, x_max, y_max, score, label, quad])

    return pred_content


gt_dict = {}  # key: img_id, value: gt object list
def parse_gt_rec(gt_filename, target_img_size, letterbox_resize=True):
    '''
    parse and re-organize the gt info.
    return:
        gt_dict: dict. Each key is a img_id, the value is the gt bboxes in the corresponding img.
    '''

    global gt_dict

    if not gt_dict:
        new_width, new_height = target_img_size
        with open(gt_filename, 'r') as f:
            for line in f:
                img_id, pic_path, boxes, labels, ori_width, ori_height, quads, img_angle = parse_line(line)

                objects = []
                for i in range(len(labels)):
                    x_min, y_min, x_max, y_max = boxes[i]
                    label = labels[i]

                    if letterbox_resize:
                        resize_ratio = min(new_width / ori_width, new_height / ori_height)

                        resize_w = int(resize_ratio * ori_width)
                        resize_h = int(resize_ratio * ori_height)

                        dw = int((new_width - resize_w) / 2)
                        dh = int((new_height - resize_h) / 2)

                        objects.append([x_min * resize_ratio + dw,
                                        y_min * resize_ratio + dh,
                                        x_max * resize_ratio + dw,
                                        y_max * resize_ratio + dh,
                                        label])
                    else:
                        objects.append([x_min * new_width / ori_width,
                                        y_min * new_height / ori_height,
                                        x_max * new_width / ori_width,
                                        y_max * new_height / ori_height,
                                        label])
                gt_dict[img_id] = objects
    return gt_dict


# The following two functions are modified from FAIR's Detectron repo to calculate mAP:
# https://github.com/facebookresearch/Detectron/blob/master/detectron/datasets/voc_eval.py
def voc_ap(rec, prec, use_07_metric=False):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))
        # print("mrec", mrec)
        # print("mpre", mpre)

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        # print("after mpre", mpre)

        # # print("mrec ", mrec)
        # # print("mpre ", mpre)
        # # plot the roc curve for the model
        # pyplot.plot(mrec, mpre)
        # show the plot
        # pyplot.show()

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

#
# def voc_eval(gt_dict, val_preds, classidx, iou_thres=0.5, use_07_metric=False):
#     '''
#     Top level function that does the PASCAL VOC evaluation.
#     '''
#     # 1.obtain gt: extract all gt objects for this class
#     class_recs = {}
#     npos = 0
#     for img_id in gt_dict:
#         R = [obj for obj in gt_dict[img_id] if obj[-1] == classidx]
#         bbox = np.array([x[:4] for x in R])
#         det = [False] * len(R)
#         npos += len(R)
#         class_recs[img_id] = {'bbox': bbox, 'det': det}
#
#     # 2. obtain pred results
#     pred = [x for x in val_preds if x[-2] == classidx]
#     img_ids = [x[0] for x in pred]
#     confidence = np.array([x[-3] for x in pred])
#     BB = np.array([[x[1], x[2], x[3], x[4]] for x in pred])
#
#     # 3. sort by confidence
#     sorted_ind = np.argsort(-confidence)
#     try:
#         BB = BB[sorted_ind, :]
#     except:
#         print('no box, ignore')
#         return 1e-6, 1e-6, 0, 0, 0
#     img_ids = [img_ids[x] for x in sorted_ind]
#
#     # 4. mark TPs and FPs
#     nd = len(img_ids)
#     tp = np.zeros(nd)
#     fp = np.zeros(nd)
#
#     for d in range(nd):
#         # all the gt info in some image
#         R = class_recs[img_ids[d]]
#         bb = BB[d, :]
#         ovmax = -np.Inf
#         BBGT = R['bbox']
#
#         if BBGT.size > 0:
#             # calc iou
#             # intersection
#             ixmin = np.maximum(BBGT[:, 0], bb[0])
#             iymin = np.maximum(BBGT[:, 1], bb[1])
#             ixmax = np.minimum(BBGT[:, 2], bb[2])
#             iymax = np.minimum(BBGT[:, 3], bb[3])
#             iw = np.maximum(ixmax - ixmin + 1., 0.)
#             ih = np.maximum(iymax - iymin + 1., 0.)
#             inters = iw * ih
#
#             # union
#             uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) + (BBGT[:, 2] - BBGT[:, 0] + 1.) * (
#                         BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)
#
#             overlaps = inters / uni
#             ovmax = np.max(overlaps)
#             jmax = np.argmax(overlaps)
#
#         if ovmax > iou_thres:
#             # gt not matched yet
#             if not R['det'][jmax]:
#                 tp[d] = 1.
#                 R['det'][jmax] = 1
#             else:
#                 fp[d] = 1.
#         else:
#             fp[d] = 1.
#
#     # compute precision recall
#     fp = np.cumsum(fp)
#     tp = np.cumsum(tp)
#     rec = tp / float(npos)
#     # avoid divide by zero in case the first detection matches a difficult
#     # ground truth
#     prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
#     ap = voc_ap(rec, prec, use_07_metric)
#
#     # return rec, prec, ap
#     return npos, nd, tp[-1] / float(npos), tp[-1] / float(nd), ap

def parse_gt_quadrangle(gt_filename, target_img_size, letterbox_resize=True):
    '''
    parse and re-organize the gt info.
    return:
        gt_dict: dict. Each key is a img_id, the value is the gt bboxes in the corresponding img.
    '''

    global gt_dict

    if not gt_dict:
        new_width, new_height = target_img_size
        with open(gt_filename, 'r') as f:
            for line in f:
                img_id, pic_path, boxes, labels, ori_width, ori_height, quads, img_angle = parse_line(line)

                objects = []
                for i in range(len(labels)):
                    x_min, y_min, x_max, y_max = boxes[i]
                    label = labels[i]
                    if label != 0:
                        continue

                    if letterbox_resize:
                        resize_ratio = min(new_width / ori_width, new_height / ori_height)

                        resize_w = int(resize_ratio * ori_width)
                        resize_h = int(resize_ratio * ori_height)

                        dw = int((new_width - resize_w) / 2)
                        dh = int((new_height - resize_h) / 2)

                        objects.append([quads[i][0] * resize_ratio + dw,
                                        quads[i][1] * resize_ratio + dh,
                                        quads[i][2] * resize_ratio + dw,
                                        quads[i][3] * resize_ratio + dh,
                                        quads[i][4] * resize_ratio + dw,
                                        quads[i][5] * resize_ratio + dh,
                                        quads[i][6] * resize_ratio + dw,
                                        quads[i][7] * resize_ratio + dh,
                                        label])
                    else:
                        objects.append([quads[i][0] * new_width / ori_width,
                                        quads[i][1] * new_height / ori_height,
                                        quads[i][2] * new_width / ori_width,
                                        quads[i][3] * new_height / ori_height,
                                        quads[i][4] * new_width / ori_width,
                                        quads[i][5] * new_height / ori_height,
                                        quads[i][6] * new_width / ori_width,
                                        quads[i][7] * new_height / ori_height,
                                        label])
                gt_dict[img_id] = objects
    return gt_dict


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return -1
       # raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

def check_in_line_segment(point, line):
    if point[0] < min(line[0][0], line[1][0]):
        return False
    elif point[0] > max(line[0][0], line[1][0]):
        return False
    elif point[1] < min(line[0][1], line[1][1]):
        return False
    elif point[1] > max(line[0][1], line[1][1]):
        return False
    else:
        return True

def calc_park_score(predict, gts, need_list = False):
    # print("predict", predict)
    points = [predict[0:2], predict[2:4], predict[4:6], predict[6:8]]
    center = np.average(points, axis=0)
    score_list = []

    for gt in gts:
        gt_points = [gt[0:2], gt[2:4], gt[4:6], gt[6:8]]
        gt_polygon = Polygon(gt_points)
        rescale_score = 1

        if not gt_polygon.contains(Point(center)):
            score_list.append(0.)
            continue

        for x, y in points:
            point = Point((x, y))
            if not gt_polygon.contains(point):
                for i in range(-1, 3):
                    intersection = line_intersection((center, (x, y)), (gt_points[i], gt_points[i + 1]))
                    if intersection == -1:
                        continue
                    if check_in_line_segment(intersection, (center, (x, y))) and check_in_line_segment(intersection, (
                    gt_points[i], gt_points[i + 1])):
                        rescale = (intersection[0] - center[0]) / (x - center[0])
                        if rescale_score > rescale:
                            rescale_score = rescale

        predict_area = Polygon(points).area
        gt_area = gt_polygon.area
        # print(rescale, np.sqrt(predict_area/gt_area) if gt_area>predict_area else np.sqrt(gt_area/predict_area))
        area_score = predict_area / gt_area if gt_area > predict_area else gt_area / predict_area
        score = area_score * rescale_score
        score_list.append(score)

    if need_list:
        if len(score_list) == 0:
            return [0]
        else:
            return score_list
    elif len(score_list) == 0:
        return 0.
    else:
        return max(score_list)

def park_eval(gt_dict, val_preds, classidx, conf_thres=0.01, score_thres=0.5, use_07_metric=False):
    '''
    Top level function that does the PASCAL VOC evaluation.
    '''

    # # Use type in predicted txt
    # f = open("E:/parking_space/data_psd_c03_1/type_gt.txt")
    # predict_dict = {}
    # lines = f.readlines()
    # for line in lines:
    #     s = line.strip().split(' ')
    #     predict_dict[int(s[0])] = int(s[1])
    # f.close()
    # pred_filtered = []
    # for x in val_preds:
    #     idx = x[0]
    #     if predict_dict[idx] != 3:
    #         pred_filtered.append(x)
    #
    # # print("val_preds", len(val_preds))
    # # print("pred_filtered", len(pred_filtered))
    # val_preds = pred_filtered

    # for img_id in gt_dict:
    #     if predict_dict[img_id] < img_ids.count(img_id):
    #         remove_num = img_ids.count(img_id) - predict_dict[img_id]
    #         indexes = [i for i,n in enumerate(img_ids) if n==img_id]
    #         for index in reversed(indexes[-remove_num:]):
    #             del img_ids[index]
    #             del pred[index]


    # 1.obtain gt: extract all gt objects for this class
    class_recs = {}
    npos = 0
    for img_id in gt_dict:
        R = [obj for obj in gt_dict[img_id] if obj[-1] == classidx]
        # bbox = np.array([x[:4] for x in R])
        quad = [x[:8] for x in R]
        det = [False] * len(R)
        npos += len(R)
        class_recs[img_id] = {'det': det, 'quad':quad}

    # 2. obtain pred results
    # pred = [x for x in val_preds if x[-2] == classidx and x[-3]>conf_thres]
    # print("val_preds", val_preds)
    pred = [x for x in val_preds if x[-2] == classidx]
    img_ids = [x[0] for x in pred]

    # Use number of empty boxes in GT
    # for img_id in gt_dict:
    #     if len(gt_dict[img_id]) < img_ids.count(img_id):
    #         remove_num = img_ids.count(img_id) - len(gt_dict[img_id])
    #         indexes = [i for i,n in enumerate(img_ids) if n==img_id]
    #         for index in reversed(indexes[-remove_num:]):
    #             del img_ids[index]
    #             del pred[index]

    # # Use number of empty boxes in predicted txt
    # f = open("E:/parking_space/data_psd_c03_1/20191104_1034#cp-0049.txt")
    # predict_dict = {}
    # lines = f.readlines()
    # for line in lines:
    #     s = line.strip().split(' ')
    #     predict_dict[int(s[0])] = int(s[1])
    # f.close()
    # for img_id in gt_dict:
    #     if predict_dict[img_id] < img_ids.count(img_id):
    #         remove_num = img_ids.count(img_id) - predict_dict[img_id]
    #         indexes = [i for i,n in enumerate(img_ids) if n==img_id]
    #         for index in reversed(indexes[-remove_num:]):
    #             del img_ids[index]
    #             del pred[index]

    score_list = np.array([calc_park_score(x[-1], class_recs[id]['quad'], need_list=True) for x, id in zip(pred, img_ids)])
    score = np.array([max(scores) for scores in score_list])
    score_top = np.array([np.argmax(scores) for scores in score_list])
    # print("score_list", len(score_list), score_list)
    # print("score", len(score), score)
    # print("score_top", len(score_top), score_top)
    # score = np.array([calc_park_score(x[-1], class_recs[id]['quad']) for x, id in zip(pred, img_ids)])
    # Quadrangle = [x[-1] for x in pred]
    # BB = np.array([[x[1], x[2], x[3], x[4]] for x in pred])
    confidence = np.array([x[-3] for x in pred])
    # print("confidence", len(confidence), confidence)

    # 3. sort by confidence
    sorted_ind = np.argsort(-confidence)
    try:
        score = score[sorted_ind]
        score_top = score_top[sorted_ind]
    except:
        print('no box, ignore')
        return 1e-6, 1e-6, 0, 0, 0
    # print("sorted_ind ", len(sorted_ind))
    # print("img_ids ", len(img_ids))

    img_ids = [img_ids[x] for x in sorted_ind]

    # for img_id in gt_dict:
    #     if img_ids.count(img_id) != len(gt_dict[img_id]):
    #         print(img_id, img_ids.count(img_id), len(gt_dict[img_id]))

    # 4. mark TPs and FPs
    nd = len(img_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    for d in range(nd):
        R = class_recs[img_ids[d]]
        # QuadGT = R['quad']
        # QuadPredict = Quadrangle[d]
        # score_list = calc_park_score(QuadPredict, QuadGT, need_list=True)

        # # all the gt info in some image
        # R = class_recs[img_ids[d]]
        # bb = BB[d, :]
        # ovmax = -np.Inf
        # BBGT = R['bbox']
        #
        # if BBGT.size > 0:
        #     # calc iou
        #     # intersection
        #     ixmin = np.maximum(BBGT[:, 0], bb[0])
        #     iymin = np.maximum(BBGT[:, 1], bb[1])
        #     ixmax = np.minimum(BBGT[:, 2], bb[2])
        #     iymax = np.minimum(BBGT[:, 3], bb[3])
        #     iw = np.maximum(ixmax - ixmin + 1., 0.)
        #     ih = np.maximum(iymax - iymin + 1., 0.)
        #     inters = iw * ih
        #
        #     # union
        #     uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) + (BBGT[:, 2] - BBGT[:, 0] + 1.) * (
        #                 BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)
        #
        #     overlaps = inters / uni
        #     ovmax = np.max(overlaps)
        #     jmax = np.argmax(overlaps)
        jmax = score_top[d]
        # print("jmax ", jmax)

        # if abs(score[d] - score_thres) < 0.01:
        #     print(score[d], img_ids[d])

        if score[d] > score_thres:
            # tp[d] = 1.
            # print(R['det'][jmax])
            # gt not matched yet
            if not R['det'][jmax]:
                tp[d] = 1.
                R['det'][jmax] = 1
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.
    # print("fp", fp)
    # print("tp", tp)

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    # print("tp", tp)
    # print("fp", fp)
    # print("rec", rec)
    # print("prec", prec)
    ap = voc_ap(rec, prec, use_07_metric)

    # return rec, prec, ap
    if len(tp) == 0:
        return npos, nd, 0., 0., ap
    return npos, nd, tp[-1] / float(npos), tp[-1] / float(nd), ap


def eval_result_file(test_file, result_file, score_th):
    gt_dict = parse_gt_quadrangle(test_file, [256, 768], False)

    result_list = []

    f = open(result_file)
    lines = f.readlines()
    f.close()

    for line in lines:
        img_id, pic_path, boxes, labels, confidences, quads = parse_result_line(line)

        for box, label, confidence, quad in zip(boxes, labels, confidences, quads):
            # quad = [box[0],box[1],box[0],box[3],box[2],box[3],box[2],box[1]] #temp test for box
            result = [img_id, box[0], box[1], box[2], box[3], confidence, label, quad]
            result_list.append(result)

    # print_result = []
    # rec_total, prec_total, fscore_total, ap_total = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    # for thresh in np.arange(0.1, 1.0, 0.1):
    npos, nd, rec, prec, ap = park_eval(gt_dict, result_list, 0, conf_thres=0.3, score_thres=score_th,
    # npos, nd, rec, prec, ap = park_eval(gt_dict, result_list, 0, conf_thres=thresh, score_thres=.8,
                                        use_07_metric=False)
    print('Recall: {:.4f}, Precision: {:.4f}, AP: {:.4f}'.format(rec, prec, ap))
    # if rec + prec == 0:
    #     fscore = .0
    # else:
    #     fscore = 2 * rec * prec / (rec + prec)
    # rec_total.update(rec)
    # prec_total.update(prec)
    # fscore_total.update(fscore)
    # ap_total.update(ap)


    # print('Recall: {:.4f}, Precision: {:.4f}, F Score: {:.4f}, AP: {:.4f}'.format(rec_total.average, prec_total.average,
    #                                                                               fscore_total.average,
    #                                                                               ap_total.average))

    # print_result.insert(0, rec_total.average)
    # print_result.insert(1, prec_total.average)
    # print_result.insert(2, fscore_total.average)
    # print_result.insert(3, ap_total.average)
    # for idx, temp in enumerate(print_result):
    #     print_result[idx] = round(temp, 4)
    # print(print_result)


def judge_correct_from_files(gt_file, result_file):
    gt_f = open(gt_file)
    result_f = open(result_file)

    gt_lines = gt_f.readlines()
    result_lines = result_f.readlines()
    gt_scores = []

    for gt_line, result_line in zip(gt_lines, result_lines):
        gts = []

        s = gt_line.strip().split(' ')
        s = s[5:] #gt에 angle 포함
        assert len(s) % 9 == 0, 'Annotation error! Please check your annotation file. Maybe partially missing some coordinates?'
        box_cnt = len(s) // 9
        for i in range(box_cnt):
            type = int(s[i * 9])
            if type == 0:
                gt_box = [float(s[i * 9 + 1]), float(s[i * 9 + 2]), float(
                s[i * 9 + 3]), float(s[i * 9 + 4]), float(s[i * 9 + 5]), float(s[i * 9 + 6]), float(
                s[i * 9 + 7]), float(s[i * 9 + 8])]
                gts.append(gt_box)
        scores = np.zeros(len(gts))

        s = result_line.strip().split(' ')
        s = s[2:]
        assert len(s) % 10 == 0, 'Annotation error! Please check your annotation file. Maybe partially missing some coordinates?'
        box_cnt = len(s) // 10
        for i in range(box_cnt):
            type = int(s[i * 10])
            if type == 0:
                predict = [float(s[i * 10 + 2]), float(s[i * 10 + 3]), float(s[i * 10 + 4]), float(s[i * 10 + 5]), float(s[i * 10 + 6]), float(
                    s[i * 10 + 7]), float(s[i * 10 + 8]), float(s[i * 10 + 9])]

                # print("predict", predict)
                # print("gts", gts)
                score_list = calc_park_score(predict, gts, need_list=True)

                scores += score_list
                # print(scores, score_list)

        for score, gt in zip(scores, gts):
            gt_scores.append([score, gt])

    return gt_scores


def calc_score_from_list(test_file, result_file):
    gt_dict = parse_gt_quadrangle(test_file, [256, 768], False)
    result_list = []

    f = open(result_file)
    lines = f.readlines()
    f.close()
    score_file = result_file.replace(".txt", "_score.txt")
    f_score = open(score_file, "wt")

    count_true = 0
    count_all = 0

    for line in lines:
        img_id, pic_path, boxes, labels, confidences, quads = parse_result_line(line)

        for box, label, confidence, quad in zip(boxes, labels, confidences, quads):
            # quad = [box[0],box[1],box[0],box[3],box[2],box[3],box[2],box[1]] #temp test for box
            result = [img_id, box[0], box[1], box[2], box[3], confidence, label, quad]
            # print(gt_dict[img_id])
            if label == 1:
                continue
            score = calc_park_score(quad, gt_dict[img_id])
            count_all += 1
            if score > 0.8:
                count_true += 1

    print(count_true, count_all)

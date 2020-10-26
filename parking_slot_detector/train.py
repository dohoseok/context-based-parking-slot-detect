import tensorflow as tf
import numpy as np
import math
# import logging
import time
import os
from tqdm import trange

# import args

from parking_slot_detector.config import *

from parking_slot_detector.utils.data_utils import get_batch_data
from parking_slot_detector.utils.misc_utils import parse_anchors, read_class_names, config_learning_rate, config_optimizer, AverageMeter
from parking_slot_detector.utils.eval_utils import evaluate_on_gpu, get_preds_gpu, park_eval, parse_gt_quadrangle
from parking_slot_detector.utils.nms_utils import gpu_nms

from parking_slot_detector.model import yolov3


def train(data_path, restore_path, save_dir, fine_tune = False):
    train_file = os.path.join(data_path, 'train.txt')
    val_file = os.path.join(data_path, 'test.txt')

    anchors = parse_anchors('data/anchors.txt')
    classes = read_class_names("data/data.names")
    class_num = len(classes)

    train_img_cnt = len(open(train_file, 'r').readlines())
    val_img_cnt = len(open(val_file, 'r').readlines())

    if fine_tune:
        batch_size = 64
        learning_rate_init = 1e-5
        restore_include = None
        restore_exclude = None
        update_part = ['yolov3/yolov3_head']
    else:
        batch_size = 20
        learning_rate_init = 1e-4
        pw_boundaries = [6., 8.]  # epoch based boundaries
        pw_values = [learning_rate_init, 3e-5, 1e-5]
        restore_include = None
        restore_exclude = ['yolov3/yolov3_head/Conv_14', 'yolov3/yolov3_head/Conv_6', 'yolov3/yolov3_head/Conv_22']
        update_part = None

    total_epoches = 11
    train_batch_num = int(math.ceil(float(train_img_cnt) / batch_size))
    train_evaluation_step = 1000
    save_epoch = 2
    warm_up_epoch = 3

    # multi_scale_train = True
    use_mix_up = True
    letterbox_resize = True

    num_threads = 10
    prefetech_buffer = 5

    use_label_smooth = True
    use_focal_loss = True
    batch_norm_decay = 0.99
    weight_decay = 5e-4
    global_step = 0

    optimizer_name = 'momentum'
    save_optimizer = True

    os.makedirs(save_dir, exist_ok=True)
    tf.reset_default_graph()

    # setting placeholders
    is_training = tf.placeholder(tf.bool, name="phase_train")
    handle_flag = tf.placeholder(tf.string, [], name='iterator_handle_flag')

    # register the gpu nms operation here for the following evaluation scheme
    pred_quads_flag = tf.placeholder(tf.float32, [1, None, None])
    pred_scores_flag = tf.placeholder(tf.float32, [1, None, None])

    gpu_nms_op = gpu_nms(pred_quads_flag, pred_scores_flag, class_num, NMS_TOPK, THRESHOLD_SCORE, THRESHOLD_NMS)

    ##################
    # tf.data pipeline
    ##################
    train_dataset = tf.data.TextLineDataset(train_file)
    train_dataset = train_dataset.shuffle(train_img_cnt)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.map(
        lambda x: tf.py_func(get_batch_data,
                             inp=[x, class_num, [INPUT_WIDTH, INPUT_HEIGHT], anchors, use_mix_up],
                             Tout=[tf.int64, tf.float32, tf.float32, tf.float32, tf.float32, tf.int64]),
        num_parallel_calls=num_threads
    )
    train_dataset = train_dataset.prefetch(prefetech_buffer)

    val_dataset = tf.data.TextLineDataset(val_file)
    val_dataset = val_dataset.batch(1)
    val_dataset = val_dataset.map(
        lambda x: tf.py_func(get_batch_data,
                             inp=[x, class_num, [INPUT_WIDTH, INPUT_HEIGHT], anchors, False],
                             Tout=[tf.int64, tf.float32, tf.float32, tf.float32, tf.float32, tf.int64]),
        num_parallel_calls=num_threads
    )
    val_dataset.prefetch(prefetech_buffer)

    iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
    train_init_op = iterator.make_initializer(train_dataset)
    val_init_op = iterator.make_initializer(val_dataset)

    # get an element from the chosen dataset iterator
    image_ids, image, y_true_13, y_true_26, y_true_52, image_angle = iterator.get_next()
    y_true = [y_true_13, y_true_26, y_true_52]

    # tf.data pipeline will lose the data `static` shape, so we need to set it manually
    image_ids.set_shape([None])
    image.set_shape([None, None, None, 3])
    for y in y_true:
        y.set_shape([None, None, None, None, None])
    image_angle.set_shape([None])
    image_angle = tf.dtypes.cast(image_angle, tf.float32)

    ##################
    # Model definition
    ##################
    yolo_model = yolov3(class_num, anchors, use_label_smooth, use_focal_loss, batch_norm_decay, weight_decay, use_static_shape=False)
    with tf.variable_scope('yolov3'):
        pred_feature_maps = yolo_model.forward(image, is_training=is_training)
    loss = yolo_model.compute_loss(pred_feature_maps, y_true, image_angle)
    y_pred = yolo_model.predict(pred_feature_maps, image_angle)

    l2_loss = tf.losses.get_regularization_loss()

    # setting restore parts and vars to updated
    saver_to_restore = tf.train.Saver(var_list=tf.contrib.framework.get_variables_to_restore(include=restore_include, exclude=restore_exclude))
    update_vars = tf.contrib.framework.get_variables_to_restore(include=update_part)

    tf.summary.scalar('train_batch_statistics/total_loss', loss[0])
    # tf.summary.scalar('train_batch_statistics/loss_xy', loss[1])
    # tf.summary.scalar('train_batch_statistics/loss_wh', loss[2])
    tf.summary.scalar('train_batch_statistics/loss_conf', loss[1])
    tf.summary.scalar('train_batch_statistics/loss_class', loss[2])
    tf.summary.scalar('train_batch_statistics/loss_quad', loss[3])
    tf.summary.scalar('train_batch_statistics/loss_l2', l2_loss)
    tf.summary.scalar('train_batch_statistics/loss_ratio', l2_loss / loss[0])

    global_step = tf.Variable(float(global_step), trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])

    if fine_tune:
        learning_rate = tf.convert_to_tensor(learning_rate_init, name='fixed_learning_rate')
    else:
        learning_rate = tf.train.piecewise_constant(global_step, boundaries=pw_boundaries, values=pw_values,
                                    name='piecewise_learning_rate')
    tf.summary.scalar('learning_rate', learning_rate)

    if not save_optimizer:
        saver_to_save = tf.train.Saver()
        saver_best = tf.train.Saver()

    optimizer = config_optimizer(optimizer_name, learning_rate)

    # set dependencies for BN ops
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        # train_op = optimizer.minimize(loss[0] + l2_loss, var_list=update_vars, global_step=global_step)
        # apply gradient clip to avoid gradient exploding
        gvs = optimizer.compute_gradients(loss[0] + l2_loss, var_list=update_vars)
        clip_grad_var = [gv if gv[0] is None else [
              tf.clip_by_norm(gv[0], 100.), gv[1]] for gv in gvs]
        train_op = optimizer.apply_gradients(clip_grad_var, global_step=global_step)

    if save_optimizer:
        print('Saving optimizer parameters to checkpoint! Remember to restore the global_step in the fine-tuning afterwards.')
        saver_to_save = tf.train.Saver()
        saver_best = tf.train.Saver()

    today = time.localtime()
    weight_folder = os.path.join(save_dir, '{:02d}{:02d}{:02d}_{:02d}{:02d}'.format(today.tm_year, today.tm_mon, today.tm_mday, today.tm_hour, today.tm_min))
    os.makedirs(weight_folder, exist_ok=True)

    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        saver_to_restore.restore(sess, restore_path)
        merged = tf.summary.merge_all()
        # writer = tf.summary.FileWriter(log_dir, sess.graph)

        print('\n----------- start to train -----------\n')

        best_mAP = -np.Inf

        for epoch in range(total_epoches):

            sess.run(train_init_op)
            loss_total, loss_conf, loss_class, loss_quad = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

            for i in trange(train_batch_num):
                _, summary, __y_pred, __y_true, __loss, __global_step, __lr = sess.run(
                    [train_op, merged, y_pred, y_true, loss, global_step, learning_rate],
                    feed_dict={is_training: True})

                # writer.add_summary(summary, global_step=__global_step)

                loss_total.update(__loss[0], len(__y_pred[0]))
                # loss_xy.update(__loss[1], len(__y_pred[0]))
                # loss_wh.update(__loss[2], len(__y_pred[0]))
                loss_conf.update(__loss[1], len(__y_pred[0]))
                loss_class.update(__loss[2], len(__y_pred[0]))
                loss_quad.update(__loss[3], len(__y_pred[0]))

                if __global_step % train_evaluation_step == 0 and __global_step > 0:
                    # recall, precision = evaluate_on_cpu(__y_pred, __y_true, class_num, nms_topk, score_threshold, nms_threshold)
                    recall, precision = evaluate_on_gpu(sess, gpu_nms_op, pred_quads_flag, pred_scores_flag, __y_pred, __y_true, class_num, THRESHOLD_NMS)

                    info = "Epoch: {}, global_step: {} | loss: total: {:.2f}, conf: {:.2f}, class: {:.2f}, quad: {:.2f} | ".format(
                            epoch, int(__global_step), loss_total.average, loss_conf.average, loss_class.average, loss_quad.average)
                    info += 'Last batch: rec: {:.3f}, prec: {:.3f} | lr: {:.5g}'.format(recall, precision, __lr)
                    print(info)
                    # logging.info(info)

                    # writer.add_summary(make_summary('evaluation/train_batch_recall', recall), global_step=__global_step)
                    # writer.add_summary(make_summary('evaluation/train_batch_precision', precision), global_step=__global_step)

                    if np.isnan(loss_total.average):
                        print('****' * 10)
                        raise ArithmeticError(
                            'Gradient exploded! Please train again and you may need modify some parameters.')

            # NOTE: this is just demo. You can set the conditions when to save the weights.
            if epoch % save_epoch == 0 and epoch > 0:
                # if loss_total.average <= 2.:
                saver_to_save.save(sess, weight_folder + '/model-epoch_{}_step_{}_loss_{:.4f}_lr_{:.5g}'.format(epoch, int(__global_step), loss_total.average, __lr))

            # switch to validation dataset for evaluation
            if epoch % save_epoch == 0 and epoch >= warm_up_epoch:
                sess.run(val_init_op)

                val_loss_total, val_loss_conf, val_loss_class, val_loss_quad = \
                    AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

                val_preds = []

                for j in trange(val_img_cnt):
                    __image_ids, __y_pred, __loss = sess.run([image_ids, y_pred, loss],
                                                             feed_dict={is_training: False})
                    pred_content = get_preds_gpu(sess, gpu_nms_op, pred_quads_flag, pred_scores_flag, __image_ids, __y_pred)
                    val_preds.extend(pred_content)
                    val_loss_total.update(__loss[0])
                    # val_loss_xy.update(__loss[1])
                    # val_loss_wh.update(__loss[2])
                    val_loss_conf.update(__loss[1])
                    val_loss_class.update(__loss[2])
                    val_loss_quad.update(__loss[3])

                # calc mAP
                rec_total, prec_total, ap_total = AverageMeter(), AverageMeter(), AverageMeter()
                # gt_dict = parse_gt_rec(val_file, [INPUT_WIDTH, INPUT_HEIGHT], letterbox_resize)
                gt_dict = parse_gt_quadrangle(val_file, [INPUT_WIDTH, INPUT_HEIGHT], letterbox_resize)

                info = '======> Epoch: {}, global_step: {}, lr: {:.6g} <======\n'.format(epoch, __global_step, __lr)

                # for ii in range(class_num):
                # for ii in range(1):
                # npos, nd, rec, prec, ap = voc_eval(gt_dict, val_preds, ii, iou_thres=eval_threshold, use_07_metric=use_voc_07_metric)
                npos, nd, rec, prec, ap = park_eval(gt_dict, val_preds, 0, conf_thres=0.3, score_thres=0.8)
                info += 'EVAL: Class {}: Recall: {:.4f}, Precision: {:.4f}, AP: {:.4f}\n'.format(0, rec, prec, ap)
                # rec_total.update(rec, npos)
                # prec_total.update(prec, nd)
                ap_total.update(ap, 1)

                mAP = ap_total.average
                # info += 'EVAL: Recall: {:.4f}, Precison: {:.4f}, mAP: {:.4f}\n'.format(rec_total.average, prec_total.average, mAP)
                info += 'EVAL: loss: total: {:.2f}, conf: {:.2f}, class: {:.2f}, quad: {:.2f}\n'.format(
                    val_loss_total.average, val_loss_conf.average, val_loss_class.average, val_loss_quad.average)
                print(info)
                # logging.info(info)

    print("weight_folder = ", weight_folder)
    trained_path = tf.train.latest_checkpoint(weight_folder)

    sess.close()

    return trained_path

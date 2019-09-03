from __future__ import print_function
import argparse
import os
import sys
import time

from PIL import Image
import tensorflow as tf
import numpy as np

from model import PLARD, PSPNet101, PSPNet50
from tools import *

SNAPSHOT_DIR = './model/ade20k_model'

ADE20k_param = {'crop_size': [473, 473],
                'num_classes': 150, # predict: [0~149] corresponding to label [1~150], ignore class 0 (background) 
                'ignore_label': 0,
                'num_steps': 2000,
                'model': PSPNet50,
                'data_dir': '../ADEChallengeData2016/', #### Change this line
                'val_list': './list/ade20k_val_list.txt'}
                
cityscapes_param = {'crop_size': [720, 720],
                    'num_classes': 19,
                    'ignore_label': 255,
                    'num_steps': 500,
                    'model': PSPNet101,
                    'data_dir': '/data/cityscapes_dataset/cityscape', #### Change this line
                    'val_list': './list/cityscapes_val_list.txt'}

kitti_pld_param = {'crop_size': [713, 713],
                   'num_classes': 2,
                   'ignore_label': 255,
                   'model': PLARD,
                   'num_steps': -1,
                   'data_dir': '/lus/scratch/aheye/data/kitti/data_road',
                   'vis_list': '',
                   'lid_dir': '/lus/scratch/aheye/data/kitti/data_road_velodyne',
                   'lid_list': ''}

kitti_psp_param = {'crop_size': [713, 713],
                   'num_classes': 19,
                   'ignore_label': 255,
                   'num_steps': -1,
                   'model': PSPNet101,
                   'data_dir': '/lus/scratch/aheye/data/kitti/data_road',
                   'vis_list': ''}

def get_arguments():
    parser = argparse.ArgumentParser(description="Reproduced PSPNet")

    parser.add_argument("--checkpoints", type=str, default=SNAPSHOT_DIR,
                        help="Path to restore weights.")
    parser.add_argument("--flipped-eval", action="store_true",
                        help="whether to evaluate with flipped img.")
    parser.add_argument("--dataset", type=str, default='',
                        choices=['ade20k', 'cityscapes', 'pld', 'psp'],
                        required=True)
    parser.add_argument("--save_dir", type=str, default='',
                        help="Path to save output.")
    parser.add_argument("--vis_dir", type=str, default=None,
                        help="Path to visual input features")
    parser.add_argument("--vis_list", type=str)
    parser.add_argument("--lid_list", type=str)
    parser.add_argument("--lid_dir", type=str, default=None,
                        help="Path to lidar input features")
    return parser.parse_args()

def load(saver, sess, ckpt_path):
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def main():
    args = get_arguments()

    # load parameters
    if args.dataset == 'ade20k':
        param = ADE20k_param
    elif args.dataset == 'cityscapes':
        param = cityscapes_param
    elif args.dataset == 'pld':
        param = kitti_pld_param
    elif args.dataset == 'psp':
        param = kitti_psp_param

    crop_size = param['crop_size']
    num_classes = param['num_classes']
    ignore_label = param['ignore_label']
    num_steps = param['num_steps']
    PSPNet = param['model']
    data_dir = param['data_dir'] if args.vis_dir is None else args.vis_dir

    # Set placeholder 
    image_filename = tf.placeholder(dtype=tf.string)
    anno_filename = tf.placeholder(dtype=tf.string)
    if args.lid_dir is not None:
        lid_filename = tf.placeholder(dtype=tf.string)
        lid = tf.cast(tf.image.decode_image(lid_filename), tf.float32)
        #lid_shape = tf.shape(lid)
        #lid_h, lid_w = (tf.minimum(713, lid_shape[0]), tf.minimum(713, lid_shape[1]))
        #lid = tf.image.crop_to_bounding_box(lid, 0, 0, lid_h, lid_w)
        lid.set_shape([713, 713, 1])
        lid = tf.expand_dims(lid, dim=0)
        print(lid)

    # Read & Decode image
    img = tf.image.decode_image(tf.read_file(image_filename), channels=3)
    anno = tf.image.decode_image(tf.read_file(anno_filename), channels=1)
    img.set_shape([None, None, 3])
    anno.set_shape([None, None, 1])

    shape = tf.shape(img)
    h, w = (tf.maximum(crop_size[0], shape[0]), tf.maximum(crop_size[1], shape[1]))
    print(h)
    print(w)
    img = preprocess(img, h, w)

     # Create network.
    if args.lid_dir is not None:
        net = PSPNet({'V_data': img, 'L_data': lid},
                     is_training=False,
                     num_classes=num_classes,
                     scale_channels=8)
    else:
        net = PSPNet({'data': img}, is_training=False, num_classes=num_classes)
   # with tf.variable_scope('', reuse=True):
   #     flipped_img = tf.image.flip_left_right(tf.squeeze(img))
   #     flipped_img = tf.expand_dims(flipped_img, dim=0)
   #     net2 = PSPNet({'data': flipped_img}, is_training=False, num_classes=num_classes)

    raw_output = net.layers['conv6']

    # Do flipped eval or not
    if args.flipped_eval:
        flipped_output = tf.image.flip_left_right(tf.squeeze(net2.layers['conv6']))
        flipped_output = tf.expand_dims(flipped_output, dim=0)
        raw_output = tf.add_n([raw_output, flipped_output])

    # Scale feature map to image size, get prediction
    raw_output_up = tf.image.resize_bilinear(raw_output, size=[h, w], align_corners=True)
    raw_output_up = tf.image.crop_to_bounding_box(raw_output_up, 0, 0, shape[0], shape[1])
    raw_output_up = tf.argmax(raw_output_up, dimension=3)
    pred = tf.expand_dims(raw_output_up, dim=3)

    # Calculate mIoU
    pred_flatten = tf.reshape(pred, [-1,])
    raw_gt = tf.reshape(anno, [-1,])
    indices = tf.squeeze(tf.where(tf.not_equal(raw_gt, ignore_label)), 1)
    gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)
    pred = tf.gather(pred_flatten, indices)

    if args.dataset == 'ade20k':
        pred = tf.add(pred, tf.constant(1, dtype=tf.int64))
        mIoU, update_op = tf.contrib.metrics.streaming_mean_iou(pred, gt, num_classes=num_classes+1)
    else:
        #road_color = np.array([255, 0, 255])
        #back_color = np.array([255, 0, 0])
        #thresh = np.array(range(0, 256))/255.0
        #gt_road = np.all(gt_image == road_color, axis=2)
        #gt_bg = np.all(gt_image == background_color, axis=2)
        #valid_gt = gt_road + gt_bg
        indices = tf.squeeze(tf.where(tf.less_equal(gt, num_classes - 1)), 1)  # ignore all labels >= num_classes
        gt = tf.cast(tf.gather(gt, indices), tf.int32)
        pred = tf.gather(pred, indices)
        mIoU, update_op = tf.contrib.metrics.streaming_mean_iou(pred, gt, num_classes=num_classes)

    # Set up tf session and initialize variables.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    global_init = tf.global_variables_initializer()
    local_init = tf.local_variables_initializer()
    sess.run(global_init)
    sess.run(local_init)

    restore_var = tf.global_variables()

    ckpt = tf.train.get_checkpoint_state(args.checkpoints)
    if ckpt and ckpt.model_checkpoint_path:
        loader = tf.train.Saver(var_list=restore_var)
        load_step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
        load(loader, sess, ckpt.model_checkpoint_path)
    else:
        print('No checkpoint file found.')
        exit() 

    img_list = open(args.vis_list, 'r')
    if args.lid_dir is not None:
        lid_list = open(args.lid_list, 'r')
    for line in img_list.readlines():
        f1, f2 = line.split(' ')
        f1 = os.path.join(data_dir, f1.strip())
        f2 = os.path.join(data_dir, f2.strip())
        if args.lid_dir is not None:
            f3 = lid_list.readline().strip()
            f3 = os.path.join(args.lid_dir, f3)
            print(f1)
            print(f2)
            print(f3)
            #_ = sess.run(update_op, feed_dict = {image_filename:f1, anno_filename:f2, lid_filename:f3})
            _ = sess.run(update_op, feed_dict = {image_filename:f1, anno_filename:f2, lid_filename:f3})
        else:
            print(f1)
            _ = sess.run(update_op, feed_dict={image_filename: f1, anno_filename: f2})

    #print('mIoU: {:04f}'.format(sess.run(mIoU)))

if __name__ == '__main__':
    main()

from __future__ import print_function

import argparse
import os
import sys
import time
import tensorflow as tf
import numpy as np
from scipy import misc
import glob

from model import PLARD, PSPNet101, PSPNet50
from tools import *

ADE20k_param = {'crop_size': [473, 473],
                'num_classes': 150, 
                'model': PSPNet50}
cityscapes_param = {'crop_size': [720, 720],
                    'num_classes': 19,
                    'model': PSPNet101}
kitti_psp_param = {'crop_size': [720, 720],
                   'num_classes': 2,
                   'model': PSPNet101}
kitti_pld_param = {'crop_size': [720, 720],
                   'num_classes': 2,
                   'model': PLARD}


SAVE_DIR = './output/'
SNAPSHOT_DIR = './model/'

def get_arguments():
    parser = argparse.ArgumentParser(description="Reproduced PSPNet")
    parser.add_argument("--checkpoints", type=str, default=SNAPSHOT_DIR,
                        help="Path to restore weights.")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR,
                        help="Path to save output.")
    parser.add_argument("--flipped-eval", action="store_true",
                        help="whether to evaluate with flipped img.")
    parser.add_argument("--dataset", type=str, default='',
                        choices=['ade20k', 'cityscapes', "PLARD", "PSPNet101"],
                        required=True)
    parser.add_argument("--vis_dir", type=str, default=None,
                        help="Path to visual input features")
    parser.add_argument("--lid_dir", type=str, default=None,
                        help="Path to lidar input features")

    return parser.parse_args()

def save(saver, sess, logdir, step):
   model_name = 'model.ckpt'
   checkpoint_path = os.path.join(logdir, model_name)

   if not os.path.exists(logdir):
      os.makedirs(logdir)
   saver.save(sess, checkpoint_path, global_step=step)
   print('The checkpoint has been created.')

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
    elif args.dataset == 'PLARD':
        param = kitti_pld_param
    elif args.dataset == 'PSPNet':
        param = kitti_psp_param

    crop_size = param['crop_size']
    num_classes = param['num_classes']
    PSPNet = param['model']

    # preprocess images
    img_files = glob.glob("%s/testing/image_2/*.png" % args.vis_dir)
    imgs = []
    for img_file in img_files:
        img, filename = load_img(img_file)
        img_shape = tf.shape(img)
        h, w = (tf.maximum(crop_size[0], img_shape[0]), tf.maximum(crop_size[1], img_shape[1]))
        imgs.append(preprocess(img, h, w))

    if args.lid_dir is not None:
        

        lids = []
        lid_files = glob.glob("%s/*.png" % args.lid_dir)
        for lid_file in lid_files:
            lid, filename = load_img(lid_file)
            lid_shape = tf.shape(lid)
            h, w = (tf.maximum(crop_size[0], lid_shape[0]), tf.maximum(crop_size[1], lid_shape[1]))
            lids.append(tf.image.decode_image(lid_file, channels=1))
        net = PSPNet({'V_data': imgs[0], 'L_data': lids[0]}, is_training=False,
                     num_classes=num_classes,
                     scale_channels=8)
    else:
        # Create network.
        net = PSPNet({'data': img}, is_training=False, num_classes=num_classes)
    #with tf.variable_scope('', reuse=True):
    #    flipped_img = tf.image.flip_left_right(tf.squeeze(img))
    #    flipped_img = tf.expand_dims(flipped_img, dim=0)
    #    net2 = PSPNet({'data': flipped_img}, is_training=False, num_classes=num_classes)

    raw_output = net.layers['conv6']
    
    # Do flipped eval or not
    if args.flipped_eval:
        flipped_output = tf.image.flip_left_right(tf.squeeze(net2.layers['conv6']))
        flipped_output = tf.expand_dims(flipped_output, dim=0)
        raw_output = tf.add_n([raw_output, flipped_output])

    # Predictions.
    raw_output_up = tf.image.resize_bilinear(raw_output, size=[h, w], align_corners=True)
    raw_output_up = tf.image.crop_to_bounding_box(raw_output_up, 0, 0, img_shape[0], img_shape[1])
    raw_output_up = tf.argmax(raw_output_up, axis=3)
    pred = decode_labels(raw_output_up, img_shape, num_classes)
    
    # Init tf Session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()

    sess.run(init)
    
    restore_var = tf.global_variables()
    
    ckpt = tf.train.get_checkpoint_state(args.checkpoints)
    if ckpt and ckpt.model_checkpoint_path:
        loader = tf.train.Saver(var_list=restore_var)
        load_step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
        load(loader, sess, ckpt.model_checkpoint_path)
    else:
        print('No checkpoint file found.')
    
    preds = sess.run(pred)
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    for pred in preds:
        misc.imsave(args.save_dir + filename, pred)
    
if __name__ == '__main__':
    main()

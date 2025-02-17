import os
import numpy as np
import tensorflow as tf

def image_mirroring(img, label, velo):
    distort_left_right_random = tf.random_uniform([1], 0, 1.0, dtype=tf.float32)[0]
    mirror = tf.less(tf.stack([1.0, distort_left_right_random, 1.0]), 0.5)
    mirror = tf.boolean_mask([0, 1, 2], mirror)
    img = tf.reverse(img, mirror)
    label = tf.reverse(label, mirror)
    velo = tf.reverse(velo, mirror)
    
    return img, label, velo

def image_scaling(img, label, velo):
    scale = tf.random_uniform([1], minval=0.5, maxval=2.0, dtype=tf.float32, seed=None)
    h_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[0]), scale))
    w_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[1]), scale))
    new_shape = tf.squeeze(tf.stack([h_new, w_new]), squeeze_dims=[1])
    img = tf.image.resize_images(img, new_shape)
    label = tf.image.resize_nearest_neighbor(tf.expand_dims(label, 0), new_shape)
    label = tf.squeeze(label, squeeze_dims=[0])
    #velo = tf.image.resize_nearest_neighbor(tf.expand_dims(velo, 0), new_shape)
    #velo = tf.squeeze(velo, squeeze_dims=[0])
    velo = tf.image.resize_images(velo, new_shape)

    return img, label, velo

def random_crop_and_pad_image_and_labels(image, label, velo, crop_h, crop_w, ignore_label=255):
    label = tf.cast(label, dtype=tf.float32)
    label = label - ignore_label # Needs to be subtracted and later added due to 0 padding.
    velo = tf.cast(velo, dtype=tf.float32)
    combined = tf.concat(axis=2, values=[image, label, velo])
    image_shape = tf.shape(image)
    combined_pad = tf.image.pad_to_bounding_box(combined, 0, 0, tf.maximum(crop_h, image_shape[0]), tf.maximum(crop_w, image_shape[1]))

    last_image_dim = tf.shape(image)[-1]
    last_label_dim = tf.shape(label)[-1] + last_image_dim
    last_velo_dim = tf.shape(velo)[-1]
    combined_crop = tf.random_crop(combined_pad, [crop_h,crop_w,4])
    img_crop = combined_crop[:, :, :3]
    label_crop = combined_crop[:, :, -2]
    label_crop = label_crop + ignore_label
    label_crop = tf.expand_dims(label_crop, dim=2)
    label_crop = tf.cast(label_crop, dtype=tf.uint8)
    velo_crop = combined_crop[:, :, -2]
    velo_crop = tf.expand_dims(velo_crop, dim=2)
    # Set static shape so that tensorflow knows shape at compile time.
    img_crop.set_shape((crop_h, crop_w, 3))
    label_crop.set_shape((crop_h,crop_w, 1))
    velo_crop.set_shape((crop_h, crop_w, 1))
    return img_crop, label_crop, velo_crop

def read_file_lists(data_dir, data_list, num_per_line):
    f = open(data_list, 'r')
    ret_lists = [[] for _ in range(num_per_line)]
    for line in f:
        try:
            splits = line[:-1].split(' ')
        except ValueError: # Adhoc for test.
            image = mask = line.strip("\n")

        if len(splits) != num_per_line:
            raise ValueError("Invalid number of paths per line!")

        paths = []
        for (i, spl) in enumerate(splits):
            path = os.path.join(data_dir, spl)
            if not tf.gfile.Exists(path):
                raise ValueError('Failed to find file' + path)
            ret_lists[i].append(path)
    return ret_lists

def read_labeled_image_list(data_dir, data_list):
    lists = read_file_lists(data_dir, data_list, 2)
    return lists[0], lists[1]

def read_lidar_list(data_dir, data_list):
    return read_file_lists(data_dir, data_list, 1)[0]

def read_images_from_disk(input_queue, input_size, random_scale, random_mirror, ignore_label, img_mean): # optional pre-processing arguments
    img_contents = tf.read_file(input_queue[0])
    label_contents = tf.read_file(input_queue[1])
    velo_contents = input_queue[2]

    img = tf.image.decode_jpeg(img_contents, channels=3)
    img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
    img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
    # Extract mean.
    img -= img_mean

    label = tf.image.decode_png(label_contents, channels=1)

    velo = tf.image.decode_image(velo_contents, channels=1)

    if input_size is not None:
        h, w = input_size

        if random_scale:
            img, label, _ = image_scaling(img, label, velo)

        if random_mirror:
            img, label, _ = image_mirroring(img, label, velo)
            
        img, label, velo = random_crop_and_pad_image_and_labels(img, label, velo, h, w, ignore_label)

    return img, label, velo

class ImageAndVeloReader(object):
    '''Generic ImageReader which reads images and corresponding segmentation
       masks from the disk, and enqueues them into a TensorFlow queue.
    '''

    def __init__(self, data_dir, data_list, 
                       velo_dir, velo_list,
                       input_size,
                       random_scale, random_mirror, 
                       ignore_label, img_mean, coord):

        self.data_dir = data_dir
        self.data_list = data_list
        self.velo_dir = velo_dir
        self.velo_list = velo_list
        self.input_size = input_size
        self.coord = coord

        self.image_list, self.label_list = read_labeled_image_list(self.data_dir, self.data_list)
        self.lidar_list = read_lidar_list(self.velo_dir, self.velo_list)
        self.images = tf.convert_to_tensor(self.image_list, dtype=tf.string)
        self.labels = tf.convert_to_tensor(self.label_list, dtype=tf.string)
        self.velos = tf.convert_to_tensor(self.lidar_list, dtype=tf.string)

        self.queue = tf.train.slice_input_producer([self.images, self.labels, self.velos],
                                                   shuffle=input_size is not None) # not shuffling if it is val
        self.image, self.label, self.velo = read_images_from_disk(self.queue, 
                                                                  self.input_size,
                                                                  random_scale, random_mirror, ignore_label, img_mean)

    def dequeue(self, num_elements):
        image_batch, label_batch, velo_batch = tf.train.batch([self.image, self.label, self.velo],
                                                  num_elements)
        return image_batch, label_batch, velo_batch

import glob
import os

input_file = "vis_train.txt"
output_file = "vis_train_pruned.txt"

in_file = open(input_file, "r")
out_file = open(output_file, "w")

for line in in_file.readlines():
  image_path, calib_path = line.split(' ')
  calib_split = calib_path.split('/')
  calib_split[1] = "gt_image_2"
  calib_split[2] = calib_split[2][:-4] + "png"
  label_path = '/'.join(calib_split).strip()
  out_file.write("%s %s\n" % (image_path, label_path))


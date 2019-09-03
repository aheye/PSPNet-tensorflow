import glob
import os

input_file = "vis_eval.txt"
output_file = "vis_eval_pruned.txt"

in_file = open(input_file, "r")
out_file = open(output_file, "w")

for line in in_file.readlines():
  pruned_line = line.split('/')[-3:]
  image_path = '/'.join(pruned_line).strip()
  pruned_line[1] = "calib"
  pruned_line[2] = pruned_line[2][:-4] + "txt"
  calib_path = '/'.join(pruned_line).strip()
  out_file.write("%s %s\n" % (image_path, calib_path))


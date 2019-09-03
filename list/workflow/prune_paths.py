import glob
import os

input_file = "lid_train.txt"
output_file = "lid_train_pruned.txt"

in_file = open(input_file, "r")
out_file = open(output_file, "w")

for line in in_file.readlines():
  filename = os.path.basename(line)
  out_file.write("%s" % filename)


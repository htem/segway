import daisy
from daisy import Coordinate, Roi
# import neuroglancer
import numpy as np
from PIL import Image
import sys
import json
import os

config_f = sys.argv[1]
with open(config_f) as f:
    config = json.load(f)
script_name = os.path.basename(config_f)
script_name = script_name.split(".")[0]

xy_downsample = 1

try:
    cutout_ds = daisy.open_ds(config["raw_file"], config["raw_ds"])
except:
    cutout_ds = daisy.open_ds(
        config["zarr"]["dir"] + "/" + script_name + ".zarr", "volumes/raw")

voxel_size = cutout_ds.voxel_size

catmaid_folder = config["CatmaidOut"].get("folder", script_name)
catmaid_f = config["CatmaidOut"]["dir"] + "/" + catmaid_folder

roi_offset = cutout_ds.roi.get_begin()
roi_shape = cutout_ds.roi.get_end() - roi_offset

# catmaid_shape = []
# for m, n in zip(cutout_ds.roi.get_shape(), voxel_size):
#     assert(m % n == 0)
#     catmaid_shape.append(m/n)
raw_dir_shape = [40, roi_shape[1], roi_shape[2]]
print(raw_dir_shape)
# exit(0)

z_begin = int(roi_offset[0] / raw_dir_shape[0])
z_end = int((roi_offset[0] + roi_shape[0]) / raw_dir_shape[0])
y_begin = int(roi_offset[1] / raw_dir_shape[1])
y_end = int((roi_offset[1] + roi_shape[1]) / raw_dir_shape[1])
if (roi_offset[1] + roi_shape[1]) % raw_dir_shape[1]:
    y_end += 1
x_begin = int(roi_offset[2] / raw_dir_shape[2])
x_end = int((roi_offset[2] + roi_shape[2]) / raw_dir_shape[2])
if (roi_offset[2] + roi_shape[2]) % raw_dir_shape[2]:
    x_end += 1

# try:
#     os.mkdir(catmaid_f)  # if errors, make sure parent folder is correct
# except:
#     pass

for z_index in range(z_begin, z_end):
    for y_index in range(y_begin, y_end):
        for x_index in range(x_begin, x_end):

            os.makedirs(catmaid_f + '/0/%d/%d' % (z_index, y_index), exist_ok=True)
            fpath = catmaid_f + '/0/%d/%d/%d.jpg' % (z_index, y_index, x_index)

            slice_roi_shape = raw_dir_shape.copy()
            x_len = raw_dir_shape[2]
            y_len = raw_dir_shape[1]
            if x_index == (x_end-1) and (roi_shape[2] % raw_dir_shape[2]):
                x_len = roi_shape[2] % raw_dir_shape[2]
                slice_roi_shape[2] = x_len
            if y_index == (y_end-1) and (roi_shape[1] % raw_dir_shape[1]):
                y_len = roi_shape[1] % raw_dir_shape[1]
                slice_roi_shape[1] = y_len

            slice_roi_offset = (z_index*raw_dir_shape[0],
                                y_index*raw_dir_shape[1],
                                x_index*raw_dir_shape[2])
            slice_roi = daisy.Roi(slice_roi_offset, slice_roi_shape)

            print(cutout_ds.roi)
            print(slice_roi)
            slice_array = cutout_ds[slice_roi].to_ndarray().reshape(
                    int(slice_roi_shape[1]/voxel_size[1]/xy_downsample),
                    int(slice_roi_shape[2]/voxel_size[2]/xy_downsample))

            # print(slice_array)

            print(fpath)
            tile = Image.fromarray(slice_array)
            tile.save(fpath, quality=95)

            continue


exit(0)

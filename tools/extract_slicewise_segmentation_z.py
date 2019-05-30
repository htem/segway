import json
import logging
import sys
import daisy
import numpy as np
from networkx import Graph
sys.path.insert(0, "/n/groups/htem/temcagt/datasets/cb2/segmentation/tri/cb2_segmentation")
from segway import task_helper

from segway.tasks.segmentation_functions import agglomerate_in_block, segment

logger = logging.getLogger(__name__)

'''
Change logs:

5/8/19: add task_config_files in config to automatically get the database address and database name

'''


def process_block(
        block,
        affs_ds,
        fragments_ds,
        # output_file,
        thresholds,
        segmentation_dss,
        ):

    total_roi = block.write_roi
    rag = Graph()
    agglomerate_in_block(
        affs_ds,
        fragments_ds,
        total_roi,
        rag
        )

    segment(
        fragments_ds,
        roi=total_roi,
        rag=rag,
        thresholds=thresholds,
        segmentation_dss=segmentation_dss
        )


def check_block(block, ds):

    # ds = daisy.open_ds(self.out_file, self.out_dataset)
    # ds = self.ds
    write_roi = ds.roi.intersect(block.write_roi)
    if write_roi.empty():
        # logger.debug("Block outside of output ROI")
        return True

    center_coord = (write_roi.get_begin() +
                    write_roi.get_end()) / 2
    center_values = ds[center_coord]
    s = np.sum(center_values)
    # logger.debug("Sum of center values in %s is %f" % (write_roi, s))

    return s != 0


if __name__ == "__main__":

    logging.basicConfig(level=logging.DEBUG)

    # try:
    #     config_f = sys.argv[1]
    #     with open(config_f) as f:
    #         config = json.load(f)
    #     file = config["affs_file"]

    # except:
    #     # try using taskhelper
    #     user_configs, global_config = task_helper.parseConfigs(sys.argv[1:])
    #     file = global_config["Input"]["output_file"]
    config_f = sys.argv[1]
    with open(config_f) as f:
        config = json.load(f)
    user_configs, global_config = task_helper.parseConfigs(config["task_config_files"])
    file = global_config["Input"]["output_file"]
    file = "/n/groups/htem/temcagt/datasets/cb2/segmentation/tri/cb2_segmentation/" + file


    affs_ds = daisy.open_ds(file, "volumes/affs")
    fragments_ds = daisy.open_ds(file, "volumes/fragments")

    num_workers = 16

    total_roi_shape = affs_ds.roi.get_shape()
    block_roi_x_dim = [x for x in total_roi_shape]
    block_roi_x_dim[2] = 80
    block_roi_x_dim = daisy.Roi((0, 0, 0), block_roi_x_dim)
    # print(block_roi_x_dim)
    block_roi_y_dim = [x for x in total_roi_shape]
    block_roi_y_dim[1] = 40
    block_roi_y_dim = daisy.Roi((0, 0, 0), block_roi_y_dim)
    # print(block_roi_y_dim)
    block_roi_z_dim = [x for x in total_roi_shape]
    block_roi_z_dim[0] = 40
    block_roi_z_dim = daisy.Roi((0, 0, 0), block_roi_z_dim)
    # print(block_roi_z_dim)

    # exit(0)

    thresholds = [.1, .15, .85, .9]
    segmentation_dss = []

    for threshold in thresholds:

        segmentation_ds = daisy.prepare_ds(
            file,
            "volumes/segmentation_slice_z" + "_%.3f" % threshold,
            fragments_ds.roi,
            fragments_ds.voxel_size,
            fragments_ds.data.dtype,
            write_roi=block_roi_z_dim,
            compressor={'id': 'zlib', 'level': 5})

        segmentation_dss.append(segmentation_ds)

    # process block-wise
    daisy.run_blockwise(
        affs_ds.roi,
        block_roi_z_dim,
        block_roi_z_dim,
        process_function=lambda b: process_block(
            b,
            affs_ds,
            fragments_ds,
            thresholds,
            segmentation_dss
            ),
        # check_function=lambda b: check_block(
        #     b, segmentation_dss[5]),
        num_workers=num_workers,
        read_write_conflict=False,
        fit='valid')

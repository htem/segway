import json
import os
import logging
import numpy as np
import sys
import daisy
# import copy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("actor_myelin_merge")


def merge_myelin_in_block(
        block,
        affs_ds,
        myelin_ds,
        merged_affs_ds,
        downsample_xy,
        low_threshold
        ):

    # assuming that myelin prediction is downsampled by an integer
    affs_array = affs_ds[block.write_roi].to_ndarray()
    myelin_array = myelin_ds[block.write_roi].to_ndarray()

    # thresholding
    np.place(myelin_array, myelin_array < low_threshold, [0])

    # up-sample myelin array
    myelin_array = np.repeat(myelin_array, downsample_xy, axis=1)
    myelin_array = np.repeat(myelin_array, downsample_xy, axis=2)
    for n, m in zip(affs_array.shape[2:3], myelin_array.shape[1:2]):
        assert(m == n)

    logger.info("Merging for ROI %s" % block.write_roi)
    ndim = 3
    for k in range(ndim):
        affs_array[k] = np.minimum(affs_array[k], myelin_array)

    # for z direction, we'd need to the section n+1 z affinity be min with the
    # section below
    # add null section for the first section
    # TODO: add context so we don't need a null section
    null_section = np.ones_like(myelin_array[0])
    null_section = 255*null_section
    myelin_array_shifted = np.concatenate([[null_section], myelin_array[:-1]])
    z_dim = 0
    affs_array[z_dim] = np.minimum(affs_array[z_dim], myelin_array_shifted)

    logger.info("Writing merged results...")
    merged_affs_ds[block.write_roi] = affs_array


if __name__ == "__main__":

    print(sys.argv)
    config_file = sys.argv[1]
    with open(config_file, 'r') as f:
        run_config = json.load(f)

    affs_file = None
    affs_dataset = None
    myelin_file = None
    myelin_dataset = None
    merged_affs_file = None
    merged_affs_dataset = None
    low_threshold = None
    downsample_xy = None

    for key in run_config:
        globals()['%s' % key] = run_config[key]

    affs_ds = daisy.open_ds(affs_file, affs_dataset)
    myelin_ds = daisy.open_ds(myelin_file, myelin_dataset)
    merged_affs_ds = daisy.open_ds(
        merged_affs_file, merged_affs_dataset, mode="r+")
    assert merged_affs_ds.data.dtype == np.uint8
    assert(low_threshold <= 255 and low_threshold >= 0)
    assert(downsample_xy is not None)

    # block_roi = affs_ds.roi
    # block = daisy.Block(block_roi, block_roi, block_roi)
    # merge_myelin_in_block(
    #     block,
    #     affs_ds,
    #     myelin_ds,
    #     merged_affs_ds)
    # exit(0)

    print("WORKER: Running with context %s" % os.environ['DAISY_CONTEXT'])
    client_scheduler = daisy.Client()

    while True:
        block = client_scheduler.acquire_block()
        if block is None:
            break

        merge_myelin_in_block(
            block,
            affs_ds,
            myelin_ds,
            merged_affs_ds,
            downsample_xy,
            low_threshold,
            )

        client_scheduler.release_block(block, ret=0)

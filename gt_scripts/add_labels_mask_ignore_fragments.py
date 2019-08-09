import daisy
import numpy as np
import sys
# import lsd
import logging
# import collections
# import multiprocessing
import scipy.ndimage as ndimage
import gt_tools

from funlib.segment.arrays import replace_values

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def __grow(array, steps):

    return ndimage.binary_dilation(array, iterations=steps)


def grow_mask(array, xy_steps, z_steps):
    assert z_steps <= xy_steps

    if z_steps:
        array = __grow(array, min(xy_steps, z_steps))

    if (xy_steps - z_steps) > 0:
        for i in range(array.shape[0]):
            array[i] = __grow(array[i], steps=xy_steps-z_steps)

    return array


if __name__ == "__main__":

    # user_configs, global_config = task_helper.parseConfigs(sys.argv[1:])

    config = gt_tools.load_config(sys.argv[1])
    file = config["file"]

    ignored_fragments = config["ignored_fragments"]

    if len(ignored_fragments) == 0:
        print("No fragments to process... Done.")
        exit(0)

    fragments_ds = daisy.open_ds(file, "volumes/fragments")
    print("Caching fragments_ds...")
    fragments_ndarray = fragments_ds.to_ndarray()

    labels_mask_ds = daisy.open_ds(file, "volumes/labels/labels_mask_z", mode='r+')
    print("Caching labels_mask_ds...")
    labels_mask_ndarray = labels_mask_ds.to_ndarray()

    # labels_mask2_ds = daisy.open_ds(file, "volumes/labels/labels_mask2")
    # print("Caching labels_mask2_ds...")
    # labels_mask2_ndarray = labels_mask2_ds.to_ndarray()

    print("Computing ignore_mask...")
    ignore_mask = np.zeros_like(fragments_ndarray, dtype=np.uint8)

    mask_values = [int(n) for n in ignored_fragments]
    new_values = [1 for n in mask_values]
    replace_values(fragments_ndarray, mask_values, new_values, ignore_mask)

    ignore_mask = grow_mask(ignore_mask, xy_steps=5, z_steps=1)

    labels_mask_ndarray[ignore_mask == 1] = 0

    print("Writing labels_mask_ds...")
    labels_mask_ds[labels_mask_ds.roi] = labels_mask_ndarray

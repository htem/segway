import daisy
# import neuroglancer
import numpy as np
from PIL import Image
import sys
import json
import lsd
import logging
import task_helper
import collections
from networkx import Graph
import threading
import multiprocessing

from task_extract_slicewise_segmentation import agglomerate_in_block

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def add_merge_error_fragments(
        fragments,
        gt,
        slice_segments,
        rag,
        ambiguous_fragments):

    fragments_by_gt = collections.defaultdict(set)
    for f in fragments:
        zyx = daisy.Coordinate(f[1])
        segid = gt[zyx]
        fragments_by_gt[segid].add(f)

    for segid in fragments_by_gt:
        # check if they all have the same id
        fragments = fragments_by_gt[segid]
        fragments_by_local_segments = collections.defaultdict(list)
        fragments_in_gt = set()
        for f in fragments:
            fragments_in_gt.add(f[0])
            zyx = daisy.Coordinate(f[1])
            local_segid = slice_segments[zyx]
            fragments_by_local_segments[local_segid].append(f)

        # for each set of fragments that do not have the same slice_id
        # get the fragment's neighbors
        # check if the neighbor belongs to any other local segments
        # if so, add both in the ignore list
        if len(fragments_by_local_segments) > 1:
            for local_segid in fragments_by_local_segments:
                local_fragments = set(
                    [f[0] for f in fragments_by_local_segments[local_segid]])
                for f in local_fragments:
                    neighbors = list(rag.adj[f].keys())
                    for n in neighbors:
                        if ((n not in local_fragments) and
                                (n in fragments_in_gt)):
                            ambiguous_fragments.add(f)
                            ambiguous_fragments.add(n)


def add_split_error_fragments(
        fragments,
        gt,
        slice_segments,
        ambiguous_fragments):

    fragments_by_slice = collections.defaultdict(set)
    for f in fragments:
        zyx = daisy.Coordinate(f[1])
        segid = slice_segments[zyx]
        fragments_by_slice[segid].add(f)

    # print(fragments)
    # print(fragments_by_slice)
    # exit(0)

    for segid in fragments_by_slice:
        # check if they all have the same id
        fragments = fragments_by_slice[segid]
        fragments_by_gt_segid = collections.defaultdict(list)
        all_fragments = set()
        for f in fragments:
            all_fragments.add(f[0])
            zyx = daisy.Coordinate(f[1])
            local_segid = gt[zyx]
            fragments_by_gt_segid[local_segid].append(f)

        # if segid == 6:
        #     print(fragments_by_gt_segid)
        #     exit(0)

        # print(fragments)
        # exit(0)
        # for each set of fragments that do not have the same slice_id
        # get the fragment's neighbors
        # check if the neighbor belongs to any other local segments
        # if so, add both in the ignore list

        # TODO: mask out internal contacts between fragments, not the whole
        # thing
        if len(fragments_by_gt_segid) > 1:
            # print(fragments_by_gt_segid)
            # exit(0)
            for gt_segid in fragments_by_gt_segid:
                gt_fragments = set(
                    [f[0] for f in fragments_by_gt_segid[gt_segid]])
                for f in gt_fragments:
                    ambiguous_fragments.add(f)


zlib_lock = multiprocessing.Lock()


def process_block(
        block,
        file,
        fragments_ds_path,
        gt_f,
        gt_ds_path,
        rag_provider,
        hi_threshold_ds,
        lo_threshold_ds,
        mask_ds,
        ):

    total_roi = block.read_roi
    print("block.read_roi: %s" % block.read_roi)
    # gt = daisy.open_ds(gt_f, gt_ds_path)[block.read_roi]
    # gt = daisy.open_ds(gt_f, gt_ds_path)

    # with zlib_lock:
    # while gt_ndarray is None:
    #     try:
    #         gt_ndarray = gt_ds[block.read_roi].to_ndarray()
    #     except:
    #         gt_ndarray = None

    print("resetting mask...")
    if reset_mask:
        mask_ds[total_roi] = 0
    fragments_ds = daisy.open_ds(file, fragments_ds_path)
    total_roi = total_roi.intersect(fragments_ds.roi)
    if total_roi.empty():
        return 0

    with zlib_lock:
        gt_ndarray = None
        while gt_ndarray is None:
            gt_ds = daisy.open_ds(gt_f, gt_ds_path)
            try:
                gt_ndarray = gt_ds[block.read_roi].to_ndarray()
            except:
                print("Failed zlib read")
                gt_ndarray = None
        print(gt_ndarray)
        gt = daisy.Array(gt_ndarray, block.read_roi, gt_ds.voxel_size)

    print("hi_threshold_ds.roi: %s" % hi_threshold_ds.roi)
    print("fragments_ds.roi: %s" % fragments_ds.roi)
    print("gt_ds.roi: %s" % gt_ds.roi)
    print("total_roi.roi: %s" % total_roi)

    rag = rag_provider[total_roi]

    all_nodes = rag.node
    fragments = []
    for n in all_nodes:
        if "center_z" in all_nodes[n]:
            f = all_nodes[n]
            fragments.append(
                (n, (f["center_z"], f["center_y"], f["center_x"])))

    ambiguous_fragments = set()

    add_merge_error_fragments(
        fragments,
        gt,
        hi_threshold_ds[total_roi],
        rag,
        ambiguous_fragments)

    add_split_error_fragments(
        fragments,
        gt,
        lo_threshold_ds[total_roi],
        ambiguous_fragments)

    print("Relabeling %s" % total_roi)
    fragments = fragments_ds[total_roi].to_ndarray()
    labels_mask = np.ones_like(fragments) * 255
    # make mask 0 for ambiguous fragments
    for f in ambiguous_fragments:
        labels_mask[fragments == f] = 0
    mask_ds[total_roi] = labels_mask


# def check_block(block, ds):

#     read_roi = ds.roi.intersect(block.read_roi)
#     if read_roi.empty():
#         return True

#     center_coord = (read_roi.get_begin() +
#                     read_roi.get_end()) / 2
#     center_values = ds[center_coord]
#     s = np.sum(center_values)

#     return s != 0


if __name__ == "__main__":

    user_configs, global_config = task_helper.parseConfigs(sys.argv[1:])

    # file = "/n/groups/htem/temcagt/datasets/cb2/segmentation/tri/cb2_segmentation/outputs/2019_02/pl2/cb2_130k/130000/output.zarr"
    file = global_config["Input"]["output_file"]
    # affs_ds = daisy.open_ds(file, "volumes/affs")
    fragments_ds = daisy.open_ds(file, "volumes/fragments")
    z_hi_threshold_ds = daisy.open_ds(file, "volumes/segmentation_slice_z_0.900")
    z_lo_threshold_ds = daisy.open_ds(file, "volumes/segmentation_slice_z_0.100")
    # x_hi_threshold_ds = daisy.open_ds(file, "volumes/segmentation_slice_80_x_0.900")
    # x_lo_threshold_ds = daisy.open_ds(file, "volumes/segmentation_slice_80_x_0.100")
    reset_mask = True
    num_workers = 8

    gt_f = "/n/groups/htem/temcagt/datasets/cb2/segmentation/segmented_volumes/cb2_pl2_181022.hdf"
    gt_ds_path = "volumes/labels/neuron_ids"
    gt_ds = daisy.open_ds(gt_f, gt_ds_path)

    rag_provider = lsd.persistence.MongoDbRagProvider(
        global_config["Input"]["db_name"],
        global_config["Input"]["db_host"],
        mode='r+',
        edges_collection="edges_" + global_config["AgglomerateTask"]["merge_function"])

    voxel_size = gt_ds.voxel_size

    context = 0  # nm
    xy_step = 40

    total_roi_offset = gt_ds.roi.get_offset()
    total_roi_shape = gt_ds.roi.get_shape()

    slice_x_shape = [x for x in total_roi_shape]
    slice_x_shape[0] = slice_x_shape[0] - 2*context
    slice_x_shape[1] = slice_x_shape[1] - 2*context
    slice_x_shape[2] = xy_step
    # slice_x_shape = daisy.Roi((context, context, 0), slice_x_shape)
    slice_x_step = daisy.Coordinate((0, 0, xy_step))
    slice_x_roi_entire = [x for x in total_roi_shape]
    slice_x_roi_entire[2] = xy_step
    slice_x_roi_entire = daisy.Roi((0, 0, 0), slice_x_roi_entire)

    z_step = 40
    slice_z_shape = [x for x in total_roi_shape]
    slice_z_shape[0] = z_step
    slice_z_shape[1] = slice_z_shape[1] - 2*context
    slice_z_shape[2] = slice_z_shape[2] - 2*context
    slice_z_roi = daisy.Roi(total_roi_offset, slice_z_shape)
    slice_z_step = daisy.Coordinate((z_step, 0, 0))
    slice_z_roi_entire = [x for x in total_roi_shape]
    slice_z_roi_entire[0] = z_step
    slice_z_roi_entire = daisy.Roi((0, 0, 0), slice_z_roi_entire)

    hi_threshold_ds = z_hi_threshold_ds
    lo_threshold_ds = z_lo_threshold_ds
    slice_roi = slice_z_roi
    slice_step = slice_z_step
    slice_roi_entire = slice_z_roi_entire

    mask_ds = daisy.prepare_ds(
        file,
        "volumes/labels/labels_mask_z",
        gt_ds.roi,
        gt_ds.voxel_size,
        np.uint8,
        write_size=slice_roi_entire.get_shape(),
        # num_channels=self.out_dims,
        # temporary fix until
        # https://github.com/zarr-developers/numcodecs/pull/87 gets approved
        # (we want gzip to be the default)
        compressor={'id': 'zlib', 'level': 5}
        )

    # print("resetting mask...")
    # if reset_mask:
    #     mask_ds[gt_ds.roi] = 0
    print("slice_roi_entire: %s" % slice_roi_entire)
    print("gt_ds.roi: %s" % gt_ds.roi)

    daisy.run_blockwise(
        gt_ds.roi,
        slice_roi_entire,
        slice_roi_entire,
        process_function=lambda b: process_block(
            b,
            file,
            "volumes/fragments",
            gt_f,
            gt_ds_path,
            rag_provider,
            hi_threshold_ds,
            lo_threshold_ds,
            mask_ds,
            ),
        # check_function=lambda b: check_block(
        #     b, segmentation_dss[5]),
        # num_workers=num_workers,
        num_workers=8,
        read_write_conflict=False,
        fit='valid')

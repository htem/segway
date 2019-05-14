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

from task_extract_slicewise_segmentation import agglomerate_in_block

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

user_configs, global_config = task_helper.parseConfigs(sys.argv[1:])


# file = "/n/groups/htem/temcagt/datasets/cb2/segmentation/tri/cb2_segmentation/outputs/2019_02/pl2/cb2_130k/130000/output.zarr"
file = global_config["Input"]["output_file"]
affs_ds = daisy.open_ds(file, "volumes/affs")
fragments_ds = daisy.open_ds(file, "volumes/fragments")
z_hi_threshold_ds = daisy.open_ds(file, "volumes/segmentation_slice_z_0.800")
z_lo_threshold_ds = daisy.open_ds(file, "volumes/segmentation_slice_z_0.200")
x_hi_threshold_ds = daisy.open_ds(file, "volumes/segmentation_slice_80_x_0.900")
x_lo_threshold_ds = daisy.open_ds(file, "volumes/segmentation_slice_80_x_0.100")

mask_ds = daisy.prepare_ds(
    file,
    "volumes/labels/labels_mask_x",
    affs_ds.roi,
    affs_ds.voxel_size,
    np.uint8,
    # write_size=raw_dir_shape,
    # num_channels=self.out_dims,
    # temporary fix until
    # https://github.com/zarr-developers/numcodecs/pull/87 gets approved
    # (we want gzip to be the default)
    compressor={'id': 'zlib', 'level': 5}
    )

f = "/n/groups/htem/temcagt/datasets/cb2/segmentation/segmented_volumes/cb2_pl2_181022.hdf"
gt_ds = daisy.open_ds(f, "volumes/labels/neuron_ids")

rag_provider = lsd.persistence.MongoDbRagProvider(
    global_config["Input"]["db_name"],
    global_config["Input"]["db_host"],
    mode='r+',
    edges_collection="edges_" + global_config["AgglomerateTask"]["merge_function"])

'''
get voxel size
for each slice in the z direction
    open the slicewise segmentation
    open database for the slice
    for each node, get xyz and get segmentation id from slice
        sort to global segmentation
        for each set, check if they have different local segmentation id
'''

voxel_size = fragments_ds.voxel_size


def add_merge_error_fragments(
        fragments,
        gt,
        slice_segments,
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


context = 200  # nm
xy_step = 80

total_roi_shape = affs_ds.roi.get_shape()
slice_x_shape = [x for x in total_roi_shape]
slice_x_shape[0] = slice_x_shape[0] - 2*context
slice_x_shape[1] = slice_x_shape[1] - 2*context
slice_x_shape[2] = xy_step
slice_x_shape = daisy.Roi((context, context, 0), slice_x_shape)
slice_x_step = daisy.Coordinate((0, 0, xy_step))

slice_x_roi_entire = [x for x in total_roi_shape]
slice_x_roi_entire[2] = xy_step
slice_x_roi_entire = daisy.Roi((0, 0, 0), slice_x_roi_entire)

hi_threshold_ds = x_hi_threshold_ds
lo_threshold_ds = x_lo_threshold_ds


slice_roi = slice_x_shape
slice_step = slice_x_step
slice_roi_entire = slice_x_roi_entire
for x in range(23, 24):
    total_roi = slice_roi.shift(slice_step * x)

    print(hi_threshold_ds.roi)
    print(gt_ds.roi)
    print(total_roi)

    gt = gt_ds[total_roi]

    rag = rag_provider[total_roi]

    all_nodes = rag.node
    # print(all_nodes)
    # exit(0)
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
        ambiguous_fragments)

    # add_split_error_fragments(
    #     fragments,
    #     gt,
    #     lo_threshold_ds[total_roi],
    #     ambiguous_fragments)

    print(total_roi)
    fragments = fragments_ds[total_roi].to_ndarray()
    mask_ds[slice_roi_entire] = 0
    labels_mask = np.ones_like(fragments) * 255
    # make mask 0 for ambiguous fragments
    for f in ambiguous_fragments:
        labels_mask[fragments == f] = 0
    mask_ds[total_roi] = labels_mask

exit(0)


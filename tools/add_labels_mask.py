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

mask_ds = daisy.prepare_ds(
    file,
    "volumes/labels/labels_mask",
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

thresholds = [.1, .2, .7, .8]
# thresholds = [.8]
segmentation_dss = []

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

voxel_size = z_hi_threshold_ds.voxel_size
context = 200  # nm


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


for z in range(80, 81):
    total_roi_shape = [40, (2000*4)-context*2, (2000*4)-context*2]
    total_roi = daisy.Roi((z*40, context, context), tuple(total_roi_shape))
    # total_roi = daisy.Roi((1180, 420, 80), (1450, 544, 80))
    # total_roi = daisy.Roi((80*40, 420*4, 1180*4), (40, (544-420)*4, (1452-1180)*4))

    entire_roi_shape = [40, 2000*4, 2000*4]
    entire_roi = daisy.Roi((z*40, 0, 0), tuple(entire_roi_shape))

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
        z_hi_threshold_ds[total_roi],
        ambiguous_fragments)

    add_split_error_fragments(
        fragments,
        gt,
        z_lo_threshold_ds[total_roi],
        # z_hi_threshold_ds[total_roi],
        ambiguous_fragments)

    print(total_roi)
    fragments = fragments_ds[total_roi].to_ndarray()
    mask_ds[entire_roi] = 0
    labels_mask = np.ones_like(fragments) * 255
    # make mask 0 for ambiguous fragments
    for f in ambiguous_fragments:
        labels_mask[fragments == f] = 0
    mask_ds[total_roi] = labels_mask

exit(0)


config_f = sys.argv[1]
with open(config_f) as f:
    config = json.load(f)

hdf_ds = daisy.open_ds(
    config['hdf']['file'],
    config['hdf']['gt'],
    )

zarr_ds = daisy.prepare_ds(
    config['zarr']['file'],
    config['zarr']['gt'],
    hdf_ds.roi,
    hdf_ds.voxel_size,
    np.uint64,
    # write_size=raw_dir_shape,
    # num_channels=self.out_dims,
    # temporary fix until
    # https://github.com/zarr-developers/numcodecs/pull/87 gets approved
    # (we want gzip to be the default)
    compressor={'id': 'zlib', 'level': 5}
    )

# exit(0)

# zarr_ds.data[:] = hdf_ds[hdf_ds.roi].to_ndarray()


exit(0)

#raw key
# gt = daisy.open_ds(f, 'volumes/labels/neuron_ids')
# lm = daisy.open_ds(f, 'volumes/labels/labels_mask')
# ul = daisy.open_ds(f, 'volumes/labels/unlabelled')

#path to labels
# f='output.zarr'


def add(s, a, name, shader=None):

    if shader == 'rgb':
        shader="""void main() { emitRGB(vec3(toNormalized(getDataValue(0)), toNormalized(getDataValue(1)), toNormalized(getDataValue(2)))); }"""

    kwargs = {}

    if shader is not None:
        kwargs['shader'] = shader

    s.layers.append(
            name=name,
            layer=neuroglancer.LocalVolume(
                data=a.data,
                offset=a.roi.get_offset()[::-1],
                voxel_size=a.voxel_size[::-1]
            ),
            **kwargs)
    print(s.layers)

viewer = neuroglancer.Viewer()

with viewer.txn() as s:

    f, prepend = (f, "test")
    #add(s, daisy.open_ds(f, 'volumes/sparse_segmentation_0.5'), '%s_seg'%prepend)
    # add(s, daisy.open_ds(f, 'volumes/affs'), '%s_aff'%prepend)
    #add(s, daisy.open_ds(f, 'volumes/fragments'), '%s_frag'%prepend)
    #add(s, daisy.open_ds(f, 'volumes/segmentation_0.100'), '%s_seg_100'%prepend)
    #add(s, daisy.open_ds(f, 'volumes/segmentation_0.200'), '%s_seg_200'%prepend)
    #add(s, daisy.open_ds(f, 'volumes/segmentation_0.300'), '%s_seg_300'%prepend)
    #add(s, daisy.open_ds(f, 'volumes/segmentation_0.400'), '%s_seg_400'%prepend)
    # add(s, daisy.open_ds(f, 'volumes/sparse_segmentation_0.5'), '%s_seg_500'%prepend)

    add(s, raw, 'raw')
    # add(s, gt, 'gt')


print(viewer)
link = str(viewer)
# print("http://catmaid3.hms.harvard.edu:%d/v/%s/" % (forwarded_port, link.split('/')[-2]))

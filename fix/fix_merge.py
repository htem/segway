import json
import logging
import daisy
import sys

import networkx
import numpy as np
import random
import collections

import task_helper

from funlib.evaluate import split_graph
from funlib.segment.arrays import replace_values

# logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.INFO)
# logging.getLogger('task_grow_segmentation').setLevel(logging.DEBUG)


def get_graph(input, threshold=0.9, rag_weight_attribute="capacity"):
    graph = networkx.Graph()
    # graph.add_nodes_from(input.nodes(data=True))
    for n, n_data in input.nodes(data=True):
        if 'center_z' in n_data:
            graph.add_node(n, **n_data)
        # if n == 58723333:
        #     print(n_data)

    # print(58723333 in graph.node)
    # print(graph.node[58723333])
    # exit(0)

    for u, v, data in input.edges(data=True):
        if u not in graph or v not in graph:
            continue
        if (data['merge_score'] is not None and
                data['merge_score'] <= threshold):
            graph.add_edge(u, v, capacity=1.0-data['merge_score'])

    return graph


def fix_merge(
        components_zyx,
        fragments_array,
        segment_array,
        rag,
        rag_weight_attribute,
        roi_offset=None,
        roi_shape=None,
        ignored_fragments=set(),
        next_segid=None,
        errored_fragments_out=None,
        **kwargs):

    # open fragments

    components_zyx = [
        [daisy.Coordinate(tuple(zyx)) for zyx in zyxs]
        for zyxs in components_zyx
    ]

    frag_to_coords = collections.defaultdict(list)
    components = []

    # preprocess data
    fragments_ids = set()
    errored_fragments = set()
    for comp_zyx in components_zyx:

        comp = []
        same_skeleton_fragments = set()

        for zyx in comp_zyx:

            if not fragments_array.roi.contains(zyx):
                print("Coord %s not in fragments_array.roi %s" % (zyx, fragments_array.roi))
                continue

            f = fragments_array[zyx]

            if f in errored_fragments:
                continue

            if f in ignored_fragments:
                # assert False, "Deprecated"
                continue

            # check for duplications within skeleton
            if f in same_skeleton_fragments:
                continue
            same_skeleton_fragments.add(f)

            # check if f does not exist in rag or filtered out
            if f not in rag.node:
                # assert 0
                continue
            frag_to_coords[f].append(zyx)

            # check if fragment is duplicated across skeletons
            if f in fragments_ids:
                print("Fragment %d is duplicated across skeletons!" % f)
                print(frag_to_coords[f])
                pixel_coords = []
                for zyx in frag_to_coords[f]:
                    print(to_pixel_coord(zyx))
                    pixel_coords.append(to_pixel_coord(zyx))

                if errored_fragments_out is None:
                    assert False
                else:
                    errored_fragments_out.append((f, pixel_coords))
                    errored_fragments.add(f)
                    # now we would need to ignore this fragments

            fragments_ids.add(f)

            # add to component
            comp.append(f)

        # add components
        if len(comp):
            components.append(comp)

    # remove problematic fragments from the component lists
    for c in components:
        for error_fragment in errored_fragments:
            try:
                c.remove(error_fragment)
            except:
                pass
    # filter out empty connected components
    components = [c for c in components if len(c)]

    # components = [
    #     [fragments_array[zyx] for zyx in comp] for comp in components_zyx]
    print("Components to be splitted %s" % components)

    # # make sure that components are of different fragments
    # for comp in components:
    #     for n in comp:
    merged_segment_id = segment_array[components_zyx[0][0]]

    if len(components) <= 1:
        return 0, merged_segment_id

    print("Input components: %s" % components)

    num_splits = split_graph(
        rag,
        components,
        position_attributes=['center_z', 'center_y', 'center_x'],
        weight_attribute=rag_weight_attribute,
        split_attribute="cut_id"
        )
    print("Num cuts made: %d" % num_splits)

    # relabeling split regions
    # assuming that all components given have the same segment_id
    parent_cut_id = rag.nodes[components[0][0]]["cut_id"]

    if next_segid is None:
        next_segid = int(random.random()*65536)

    # create new segment IDs
    new_segment_ids = {}
    for i in range(num_splits+1):
        if i == parent_cut_id:
            new_segment_ids[i] = merged_segment_id
        else:
            new_segment_ids[i] = next_segid
            next_segid += 1

    # create remap list
    mask_values = []
    new_values = []

    rewrite_segment = True
    # rewrite_segment = False

    if rewrite_segment:
        print("Computing new segmentation...")
        for node, node_data in rag.nodes(data=True):
            node_zyx = daisy.Coordinate(tuple((node_data['center_z'],
                                        node_data['center_y'],
                                        node_data['center_x'])))
            if segment_array[node_zyx] == merged_segment_id:
                if node_data["cut_id"] != parent_cut_id:
                    mask_values.append(node)
                    new_values.append(new_segment_ids[node_data["cut_id"]])

        mask_values = np.array(mask_values, dtype=fragments_array.dtype)
        new_values = np.array(new_values, dtype=segment_array.dtype)

        print("Replacing values...")
        segment_ndarray = segment_array.to_ndarray()
        replace_values(
            fragments_array.to_ndarray(),
            mask_values,
            new_values,
            segment_ndarray,
            )
        print("Write new segmentation...")
        segment_array[segment_array.roi] = segment_ndarray

        return num_splits, merged_segment_id


def to_daisy_coord(xyz):
    return [xyz[2]*40, xyz[1]*4, xyz[0]*4]


def to_pixel_coord(zyx):
    return [zyx[2]/4, zyx[1]/4, zyx[0]/40]


if __name__ == "__main__":
    '''Test case for fixing splitter'''

    pass

    # user_configs, global_config = task_helper.parseConfigs(sys.argv[1:])

    # components_zyx = []
    # components_zyx.append([[916, 1121, 147]])
    # components_zyx.append([[917, 1247, 147]])
    # # components_zyx.append([[1325, 1243, 111]])
    # # components_zyx.append([[920, 790, 73]])
    # # tests = [[[580, 576, 106]], [[935, 960, 106]]]

    # components_zyx = [[to_daisy_coord(n) for n in c] for c in components_zyx]
    # print(components_zyx)

    # fragments_array = daisy.open_ds(fragments_file, fragments_dataset)
    # print("Making fragment cache...")
    # fragments_cached = fragments_array[fragments_array.roi].to_ndarray()
    # fragments_cached = daisy.Array(
    #     fragments_cached, fragments_array.roi, fragments_array.voxel_size)
    # # fragments_cached = fragments_array

    # segment_dataset = "volumes/segmentation_0.900"
    # segment_threshold = float(segment_dataset.split('_')[-1])

    # segment_ds = daisy.open_ds(segment_file, segment_dataset, mode='r+')
    # print("Making segment cache...")
    # segment_ndarray = segment_ds[segment_ds.roi].to_ndarray()
    # segment_ds_cached = daisy.Array(
    #     segment_ds_cached, segment_ds.roi, segment_ds.voxel_size)
    # segment_array = daisy.Array(
    #     segment_ndarray, total_roi, fragments_array.voxel_size)
    # # open RAG DB
    # rag_provider = daisy.persistence.MongoDbGraphProvider(
    #     db_name,
    #     host=db_host,
    #     mode='r',
    #     edges_collection=edges_collection,
    #     position_attribute=['center_z', 'center_y', 'center_x'])

    # subrag = rag_provider[total_roi]
    # rag = get_graph(subrag, segment_threshold)

    # if roi_offset is not None and roi_shape is not None:
    #     assert(total_roi is not None)
    # else:
    #     total_roi = daisy.Roi(tuple(roi_offset), tuple(roi_shape))

    # fix_merge(
    #     components_zyx=components_zyx,
    #     segment_dataset=segment_dataset,
    #     segment_threshold=segment_threshold,
    #     segment_ndarray=None,
    #     **user_configs, **global_config["FixMergeTask"])

    # if segment_ds is not None:
    #     # rewrite segment_file
    #     print("Writing new segmentation file...")
    #     segment_ds[segment_ds.roi] = new_segmentation
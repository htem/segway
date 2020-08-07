# import json
import logging
import daisy
# import sys
import networkx
import numpy as np
# import rando# import json
# import logging
import daisy
# import sys
import networkx
import numpy as np
# import random
import collections

from segway.tasks.segmentation_functions import agglomerate_in_block, segment

# from segway import task_helper

# from funlib.evaluate import split_graph
# from funlib.segment.arrays import replace_values

# logging.basicConfig(level=logging.INFO)


# def get_graph(input, threshold=0.9, rag_weight_attribute="capacity"):
#     graph = networkx.Graph()
#     for n, n_data in input.nodes(data=True):
#         if 'center_z' in n_data:
#             graph.add_node(n, **n_data)

#     for u, v, data in input.edges(data=True):
#         if u not in graph or v not in graph:
#             continue
#         if (data['merge_score'] is not None and
#                 data['merge_score'] <= threshold):
#             graph.add_edge(u, v, capacity=1.0-data['merge_score'])

#     return graph


# def get_segment_id(zyx, fragments_array, fragments_lut, segment_array):
#     fid = fragments_array[zyx]
#     if fragments_lut is not None and fid in fragments_lut:
#         ret = fragments_lut[fid]
#     else:
#         ret = segment_array[zyx]
#     assert ret is not None
#     return ret


def fix_merge2(
        components_zyx,
        affs_array,
        fragments_array,
        segment_array,
        # rag,
        # rag_weight_attribute,
        roi_offset=None,
        roi_shape=None,
        ignored_fragments=set(),
        next_segid=None,
        errored_fragments_out=None,
        # fragments_lut=None,
        # roi=None,
        reagglomeration_threshold=0.7,
        ):

    if roi_offset is None or roi_shape is None:
        assert roi_offset is None
        assert roi_shape is None
        roi = affs_array.roi
    else:
        roi = daisy.Roi(roi_offset, roi_shape)
    assert affs_array.roi.contains(roi)
    assert fragments_array.roi.contains(roi)

    # make sure coords are `Coordinate`s
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

            if not roi.contains(zyx):
                # print("Coord %s not in fragments_array.roi %s" % (zyx, roi))
                continue

            f = fragments_array[zyx]

            if f in errored_fragments:
                continue

            if f in ignored_fragments:
                continue

            # check for duplications within skeleton
            if f in same_skeleton_fragments:
                continue
            same_skeleton_fragments.add(f)
            frag_to_coords[f].append(zyx)

            # check if fragment is duplicated across skeletons
            if f in fragments_ids:
                # print("Fragment %d is duplicated across skeletons!" % f)
                # print(frag_to_coords[f])
                pixel_coords = []
                for zyx in frag_to_coords[f]:
                    # print(to_pixel_coord(zyx))
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

    # remove problem fragments from the component lists
    for c in components:
        for error_fragment in errored_fragments:
            try:
                c.remove(error_fragment)
            except:
                pass
    # filter out empty connected components
    components = [c for c in components if len(c)]

    if len(components) <= 1:
        return

    # print(components); exit()

    rag = networkx.Graph()
    agglomerate_in_block(
        affs_array,
        fragments_array,
        roi,
        rag,
        unmerge_list=[components]
        )

    # exit()

    segment(
        fragments_array,
        roi=roi,
        rag=rag,
        thresholds=[reagglomeration_threshold],
        segmentation_dss=[segment_array]
        )

    return


def to_daisy_coord(xyz):
    return [xyz[2]*40, xyz[1]*4, xyz[0]*4]


def to_pixel_coord(zyx):
    return [zyx[2]/4, zyx[1]/4, zyx[0]/40]


if __name__ == "__main__":
    '''Test case for fixing splitter'''

    pass

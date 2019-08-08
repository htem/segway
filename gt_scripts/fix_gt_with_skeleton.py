import json
import logging
import daisy
import sys
import math
import collections
# import numpy as np
import gt_tools

from funlib.segment.arrays import replace_values

from fix_merge import get_graph, fix_merge

try:
    import graph_tool
except ImportError:
    print("Error: graph_tool is not found.")
    exit(0)

logging.basicConfig(level=logging.INFO)


def to_daisy_coord(xyz):
    return [xyz[2]*40, xyz[1]*4, xyz[0]*4]


def to_pixel_coord(zyx):
    return [zyx[2]/4, zyx[1]/4, zyx[0]/40]


def to_zyx(xyz):
    return [xyz[2], xyz[1], xyz[0]]


def segment_from_skeleton(skeletons, segment_array, nodes):
    segments = collections.defaultdict(lambda: collections.defaultdict(list))

    for skeleton_id in skeletons:
        for n in skeletons[skeleton_id]:
            zyx = daisy.Coordinate(tuple(nodes[n]["zyx"]))
            if not segment_array.roi.contains(zyx):
                continue
            segid = segment_array[zyx]
            assert(segid is not None)
            segments[segid][skeleton_id].append(n)

    return segments


def get_one_merged_components(segments, done):
    for s in segments:
        if s in done:
            continue
        components = segments[s]
        if len(components) > 1:
            return [components[c] for c in components]
    return None


def get_one_splitted_component(skeletons, segment_array, nodes, done):
    for skid in skeletons:
        segments = set()
        zyxs = []
        if skid in done:
            continue
        s = skeletons[skid]
        for n in s:
            zyx = daisy.Coordinate(tuple(nodes[n]["zyx"]))
            if not segment_array.roi.contains(zyx):
                continue
            seg_id = segment_array[zyx]
            if seg_id != 0:
                # TODO: not entirely sure why this could be
                # when bound is checked above
                segments.add(seg_id)
                zyxs.append(zyx)
        if len(segments) > 1:
            return (segments, skid, zyxs)
    return (None, None, None)


def interpolate_locations_in_z(zyx0, zyx1):
    # assuming z pixel size is 40
    assert(zyx0[0] % 40 == 0)
    assert(zyx1[0] % 40 == 0)
    steps = int(math.fabs(zyx1[0] - zyx0[0]) / 40)
    if steps <= 1:
        return []

    delta = []
    for i in range(3):
        delta.append((float(zyx1[i]) - zyx0[i]) / steps)
    assert(int(delta[0]) == 40 or int(delta[0]) == -40)
    res = []
    for i in range(steps-1):
        res.append([int(zyx0[k] + (i+1)*delta[k]) for k in range(3)])

    return res


def parse_skeleton_json(json, interpolation=True):
    skeletons = {}
    nodes = {}
    json = json["skeletons"]
    for skel_id in json:
        skel_json = json[skel_id]
        # skeletons[skel_id] = []
        skeleton = []
        for node_id_json in skel_json["treenodes"]:
            node_json = skel_json["treenodes"][node_id_json]
            node = {"zyx": to_zyx(node_json["location"])}
            node_id = len(nodes)
            nodes[node_id] = node

            if interpolation:
                if node_json["parent_id"] is not None:
                    # need to check and make intermediate nodes
                    # before appending current node
                    prev_node_id = str(node_json["parent_id"])
                    prev_node = skel_json["treenodes"][prev_node_id]
                    intermediates = interpolate_locations_in_z(
                        to_zyx(prev_node["location"]), node["zyx"])
                    for loc in intermediates:
                        int_node_id = len(nodes)
                        int_node = {"zyx": loc}
                        nodes[int_node_id] = int_node
                        skeleton.append(int_node_id)

            skeleton.append(node_id)

        if len(skeleton) == 1:
            # skip single node skeletons (likely to be synapses annotation)
            continue
        
        skeletons[skel_id] = skeleton
    return skeletons, nodes


if __name__ == "__main__":

    config = gt_tools.load_config(sys.argv[1])
    file = config["file"]

    update_existing_volume = False
    if len(sys.argv) > 2 and sys.argv[2] == "--update":
        update_existing_volume = True

    correct_merges = True
    if len(sys.argv) > 2 and sys.argv[2] == "--no_correct_merge":
        correct_merges = False
    correct_splits = True
    if len(sys.argv) > 2 and sys.argv[2] == "--no_correct_split":
        correct_splits = False
    make_segment_cache = True
    make_fragment_cache = True

    skeleton_json = config["skeleton_file"]
    with open(skeleton_json) as f:
        skeletons, nodes = parse_skeleton_json(json.load(f), interpolation=True)

    segmentation_skeleton_ds = config["segmentation_skeleton_ds"]

    if update_existing_volume:
        segment_dataset = segmentation_skeleton_ds
    else:
        segment_dataset = config["segment_ds"]

    segment_file = config.get("segment_file", config["file"])
    segment_ds = daisy.open_ds(segment_file, segment_dataset)
    segment_array = segment_ds[segment_ds.roi]
    if make_segment_cache:
        print("Making segment cache...")
        segment_ndarray = segment_array.to_ndarray()
        segment_array = daisy.Array(
            segment_ndarray, segment_ds.roi, segment_ds.voxel_size)

    segment_threshold = float(config["segment_ds"].split('_')[-1])

    fragments_file = config.get("fragments_file", config["file"])
    fragments_dataset = config.get("fragments_ds", 'volumes/fragments')
    fragments_ds = daisy.open_ds(fragments_file, fragments_dataset)
    fragments_array = fragments_ds[fragments_ds.roi]

    print("Creating corrected segment at %s" % segmentation_skeleton_ds)

    corrected_segment_ds = daisy.prepare_ds(
        segment_file,
        segmentation_skeleton_ds,
        segment_ds.roi,
        segment_ds.voxel_size,
        segment_ds.dtype,
        compressor={'id': 'zlib', 'level': 5}
        )

    if correct_merges:

        if make_fragment_cache:
            print("Making fragment cache...")
            fragments_array.materialize()

        print("Making rag cache...")
        # open RAG DB
        position_attributes = ['center_z', 'center_y', 'center_x']
        rag_provider = daisy.persistence.MongoDbGraphProvider(
            config["db_name"],
            host=config["db_host"],
            mode='r',
            edges_collection=config["db_edges_collection"],
            position_attribute=position_attributes)
        subrag = rag_provider[fragments_ds.roi]
        # sanity check
        assert len(subrag.nodes) > 0
        assert len(subrag.edges) > 0

        rag_weight_attribute = "capacity"
        rag = get_graph(subrag, segment_threshold, rag_weight_attribute)

        # sanity check
        assert len(rag.nodes) > 0
        assert len(rag.edges) > 0

        print("Get max segment_id...")
        max_segid = 0
        for n, n_data in rag.nodes(data=True):
            zyx = daisy.Coordinate(tuple([n_data[c] for c in position_attributes]))
            max_segid = max(segment_array[zyx], max_segid)
        assert max_segid != 0
        next_segid = max_segid + 1

        # for known fragments that have nodes of multiple skeletons, we will have to assign them unique IDs, and disable their masks at a later step
        ignored_fragments = []
        if "ignored_fragments" in config:
            print("Processing ignored_fragments...")
            ignored_fragments = config["ignored_fragments"]
            mask_values = []
            new_values = []
            for f in ignored_fragments:
                mask_values.append(f)
                new_values.append(next_segid)
                next_segid += 1

            segment_ndarray = segment_array.to_ndarray()
            replace_values(
                segment_ndarray,
                mask_values,
                new_values,
                segment_ndarray,
                )
            segment_array[segment_array.roi] = segment_ndarray

        print("Correcting merges...")
        processed_segments = set()
        errored_fragments = []
        while True:

            print("Getting segments from skeletons...")
            segment_by_skeletons = segment_from_skeleton(
                skeletons, segment_array, nodes)

            merge_components = get_one_merged_components(
                    segment_by_skeletons, processed_segments)

            if merge_components is None:
                print("No more merged components found")
                break

            # convert to zyx
            components_zyx = [
                [nodes[n]["zyx"] for n in c] for c in merge_components]

            print("Fix merged components...")
            for c in components_zyx:
                print([to_pixel_coord(node_zyx) for node_zyx in c])

            n_splits, segment_id = fix_merge(
                components_zyx,
                fragments_array,
                segment_array,
                rag,
                rag_weight_attribute,
                ignored_fragments=ignored_fragments,
                next_segid=next_segid,
                errored_fragments_out=errored_fragments,
                )

            next_segid += n_splits
            processed_segments.add(segment_id)

        if len(errored_fragments):
            for f in errored_fragments:
                print(f)
            assert False, "There are error fragments, put them in ignored_fragments first"

    if correct_splits:
        processed_segments = set()

        while True:
            splitted_segments, skeleton_id, coords = get_one_splitted_component(
                    skeletons,
                    segment_array,
                    nodes,
                    processed_segments)
            processed_segments.add(skeleton_id)

            if splitted_segments is None:
                break  # done

            print("Splitted segments:")
            for s, zyx in zip(splitted_segments, coords):
                # print("Splitted segments: %s" % splitted_segments)
                print("%s (%s)" % (s, to_pixel_coord(zyx)), end=', ')

            mask_values = list(splitted_segments)
            new_values = [mask_values[0] for k in mask_values]

            print("Replacing values...")
            segment_ndarray = segment_array.to_ndarray()
            replace_values(
                segment_ndarray,
                mask_values,
                new_values,
                segment_ndarray,
                )
            print("Write new segmentation...")
            segment_array[segment_array.roi] = segment_ndarray

    print("Write segmentation to disk...")
    corrected_segment_ds[corrected_segment_ds.roi] = segment_array.to_ndarray()

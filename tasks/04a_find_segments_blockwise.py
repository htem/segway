import daisy
import json
import logging
import sys
import time
import os

import pymongo
import numpy as np
# import multiprocessing as mp
from funlib.segment.graphs.impl import connected_components

import task_helper

logging.basicConfig(level=logging.INFO)
logging.getLogger('daisy.persistence.shared_graph_provider').setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)
# np.set_printoptions(threshold=sys.maxsize, formatter={'all':lambda x: str(x)})


def find_segments(
        db_host,
        db_name,
        fragments_file,
        lut_dir,
        edges_collection,
        merge_function,
        roi_offset,
        roi_shape,
        thresholds,
        run_type=None,
        block_id=None,
        **kwargs):

    '''

    Args:

        db_host (``string``):

            Where to find the MongoDB server.

        db_name (``string``):

            The name of the MongoDB database to use.

        fragments_file (``string``):

            Path to the file containing the fragments.

        edges_collection (``string``):

            The name of the MongoDB database collection to use.

        roi_offset (array-like of ``int``):

            The starting point (inclusive) of the ROI. Entries can be ``None``
            to indicate unboundedness.

        roi_shape (array-like of ``int``):

            The shape of the ROI. Entries can be ``None`` to indicate
            unboundedness.

    '''

    print("Reading graph from DB ", db_name, edges_collection)
    start = time.time()

    graph_provider = daisy.persistence.MongoDbGraphProvider(
        db_name,
        db_host,
        edges_collection=edges_collection,
        position_attribute=[
            'center_z',
            'center_y',
            'center_x'])

    roi = daisy.Roi(
        roi_offset,
        roi_shape)

    node_attrs, edge_attrs = graph_provider.read_blockwise(
        roi,
        block_size=daisy.Coordinate((8000, 8192, 8192)),
        num_workers=1)
    # node_attrs, edge_attrs = graph_provider.read_block(
    #     roi,
    #     block_size=daisy.Coordinate((8000, 8192, 8192)),
    #     num_workers=4)

    print("Read graph in %.3fs" % (time.time() - start))

    if 'id' not in node_attrs:
        print('No nodes found in roi %s' % roi)
        return

    print('id dtype: ', node_attrs['id'].dtype)
    print('edge u  dtype: ', edge_attrs['u'].dtype)
    print('edge v  dtype: ', edge_attrs['v'].dtype)

    nodes = node_attrs['id']
    u_array = edge_attrs['u'].astype(np.uint64)
    v_array = edge_attrs['v'].astype(np.uint64)
    edges = np.stack([u_array, v_array], axis=1)

    if len(u_array) == 0:
        # this block somehow has no edges, or is not agglomerated
        u_array = np.array([0], dtype=np.uint64)
        v_array = np.array([0], dtype=np.uint64)
        edges = np.array([[0, 0]], dtype=np.uint64)

        # assert False, "Block %s is empty!"

    if 'merge_score' in edge_attrs:
        scores = edge_attrs['merge_score'].astype(np.float32)
    else:
        scores = np.ones_like(u_array, dtype=np.float32)

    # for i in range(len(scores)):
    #     print("%d to %d: %f" % (u_array[i], v_array[i], scores[i]))

    print('Nodes dtype: ', nodes.dtype)
    print('edges dtype: ', edges.dtype)
    print('scores dtype: ', scores.dtype)

    # each block should have at least one node, edge, and score
    assert len(nodes)
    assert len(edges)
    assert len(scores)

    # print("edge_attrs: ", edge_attrs)

    print("Complete RAG contains %d nodes, %d edges" % (len(nodes), len(edges)))

    out_dir = os.path.join(
        fragments_file,
        lut_dir)

    if run_type:
        out_dir = os.path.join(out_dir, run_type)

    os.makedirs(out_dir, exist_ok=True)

    start = time.time()

    for threshold in thresholds:

        get_connected_components(
                nodes,
                # u_array,
                # v_array,
                edges,
                scores,
                threshold,
                merge_function,
                out_dir,
                block_id)

        print("Created and stored lookup tables in %.3fs" % (time.time() - start))

def get_connected_components(
        nodes,
        # u_array,
        # v_array,
        edges,
        scores,
        threshold,
        merge_function,
        out_dir,
        block_id=None,
        hi_threshold=0.95,
        **kwargs):

    if block_id is None:
        block_id = 0

    print("Getting CCs for threshold %.3f..." % threshold)
    start = time.time()

    edges_tmp = edges[scores <= threshold]
    scores_tmp = scores[scores <= threshold]

    if len(edges_tmp) == 0:
        print("edges_tmp: ", edges_tmp)
        print("scores_tmp: ", scores_tmp)
        print("edges: ", edges)
        print("scores: ", scores)

    components = connected_components(nodes, edges_tmp, scores_tmp, threshold,
                                      use_node_id_as_component_id=1)
    print("%.3fs" % (time.time() - start))

    print("Creating fragment-segment LUT for threshold %.3f..." % threshold)
    start = time.time()
    lut = np.array([nodes, components])

    print("%.3fs" % (time.time() - start))

    print("Storing fragment-segment LUT for threshold %.3f..." % threshold)
    start = time.time()

    logger.info("Block: %s" % block)

    # print("******Local LUT: ")
    # for i in range(len(lut[0])):
    #     print("%d: %d" % (lut[0][i], lut[1][i]))


    lookup = 'seg_frags2local_%s_%d/%d' % (merge_function, int(threshold*100), block_id)
    out_file = os.path.join(out_dir, lookup)
    np.savez_compressed(out_file, fragment_segment_lut=lut)

    unique_components = np.unique(components)
    lookup = 'nodes_%s_%d/%d' % (merge_function, int(threshold*100), block_id)
    out_file = os.path.join(out_dir, lookup)
    np.savez_compressed(out_file, nodes=unique_components)

    # print("Num components: ", len(unique_components))
    # # print(nodes_in_vol)
    # print("Num nodes in vol: ", len(nodes_in_vol))

    nodes_in_vol = set(nodes)

    def not_in_graph(u, v):
        return u not in nodes_in_vol or v not in nodes_in_vol

    print("Num edges original: ", len(edges))
    outward_edges = np.array([not_in_graph(n[0], n[1]) for n in edges])
    edges = edges[np.logical_and(scores <= threshold, outward_edges)]

    # print("Local-frag edges: ", edges)

    # replace IDs in edges with agglomerated IDs
    frags2seg = {n: k for n, k in np.dstack((lut[0], lut[1]))[0]}
    for i in range(len(edges)):
        if edges[i][0] in frags2seg:
            if edges[i][0] != frags2seg[edges[i][0]]:
                edges[i][0] = frags2seg[edges[i][0]]
        if edges[i][1] in frags2seg:
            if edges[i][1] != frags2seg[edges[i][1]]:
                edges[i][1] = frags2seg[edges[i][1]]

    if len(edges):
        # np.unique doesn't work on empty arrays
        edges = np.unique(edges, axis=0)

    print("Num edges pruned: ", len(edges))

    # edges = edges[np.logical_or(scores >= threshold, outward_edges)]
    # scores = scores[np.logical_or(scores >= threshold, outward_edges)]
    # # edges = edges[np.logical_or(scores >= threshold, outward_edges)]
    # # print(np.stack([u_array, v_array, scores, outward_edges], axis=1))
    # # print("Num out edges left: ", len(outward_edges))
    # # print(edges)
    # # print(scores)
    # # print()

    # edges = edges[np.logical_or(scores >= threshold, outward_edges)]
    # scores = scores[np.logical_or(scores >= threshold, outward_edges)]
    # print("Num edges: ", len(edges))

    # # prune _all_ edges with scores higher than hi_threshold
    # edges = edges[scores < hi_threshold]
    # print("Num edges: ", len(edges))

    lookup = 'edges_local2frags_%s_%d/%d' % (merge_function, int(threshold*100), block_id)
    out_file = os.path.join(out_dir, lookup)
    np.savez_compressed(out_file, edges=edges)

    print("%.3fs" % (time.time() - start))


if __name__ == "__main__":

    if sys.argv[1] == 'run':

        user_configs, global_config = task_helper.parseConfigs(sys.argv[2:])
        config = user_configs
        config.update(global_config["FindSegmentsBlockwiseTask"])

        find_segments(**config)

    else:

        print(sys.argv)
        config_file = sys.argv[1]
        with open(config_file, 'r') as f:
            run_config = json.load(f)

        for key in run_config:
            globals()['%s' % key] = run_config[key]

        print("WORKER: Running with context %s"%os.environ['DAISY_CONTEXT'])
        client_scheduler = daisy.Client()

        db_client = pymongo.MongoClient(db_host)
        db = db_client[db_name]
        completion_db = db[completion_db_name]

        while True:
            block = client_scheduler.acquire_block()
            if block is None:
                break

            roi_offset = block.write_roi.get_offset()
            roi_shape = block.write_roi.get_shape()

            find_segments(
                db_host=db_host,
                db_name=db_name,
                fragments_file=fragments_file,
                lut_dir=lut_dir,
                edges_collection=edges_collection,
                merge_function=merge_function,
                roi_offset=roi_offset,
                roi_shape=roi_shape,
                thresholds=thresholds,
                block_id=block.block_id,
                )

            # recording block done in the database
            document = dict()
            document.update({
                'block_id': block.block_id,
                'read_roi': (block.read_roi.get_begin(), block.read_roi.get_shape()),
                'write_roi': (block.write_roi.get_begin(), block.write_roi.get_shape()),
                'start': 0,
                'duration': 0
            })
            completion_db.insert(document)

            client_scheduler.release_block(block, ret=0)

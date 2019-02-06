import json
import logging
import lsd
# import os
import daisy
import sys
import datetime
import time

import networkx
# from parallel_read_rag import parallel_read_rag

import task_helper
from parallel_relabel import parallel_relabel

logging.basicConfig(level=logging.INFO)
# logging.getLogger('lsd.persistence.mongodb_rag_provider').setLevel(logging.DEBUG)


def get_connected_components(graph, threshold):
    '''Get all connected components in the RAG, as indicated by the
    'merge_score' attribute of edges.'''

    merge_graph = networkx.Graph()
    merge_graph.add_nodes_from(graph.nodes())

    for u, v, data in graph.edges(data=True):
        if (data['merge_score'] is not None and
                data['merge_score'] <= threshold):
            merge_graph.add_edge(u, v)

    components = networkx.connected_components(merge_graph)

    return [list(component) for component in components]


def extract_segmentation(
        fragments_file,
        fragments_dataset,
        out_file,
        out_dataset,
        db_host,
        db_name,
        edges_collection,
        thresholds,
        roi_offset=None,
        roi_shape=None,
        num_workers=4,
        **kwargs):

    print(datetime.datetime.now())

    # open fragments
    fragments = daisy.open_ds(fragments_file, fragments_dataset)

    # open RAG DB
    rag_provider = daisy.persistence.MongoDbGraphProvider(
        db_name,
        host=db_host,
        mode='r',
        edges_collection=edges_collection,
        position_attribute=['center_z', 'center_y', 'center_x'])

    # # open RAG DB
    # rag_provider = lsd.persistence.MongoDbRagProvider(
    #     db_name,
    #     host=db_host,
    #     mode='r',
    #     edges_collection=edges_collection)

    total_roi = fragments.roi
    if roi_offset is not None:
        assert roi_shape is not None, "If roi_offset is set, roi_shape " \
                                      "also needs to be provided"
        total_roi = daisy.Roi(offset=roi_offset, shape=roi_shape)

    print(datetime.datetime.now())
    print("Reading RAG in %s" % total_roi)
    rag = rag_provider[total_roi]
    print("Number of nodes in RAG: %d" % (len(rag.nodes())))
    print("Number of edges in RAG: %d" % (len(rag.edges())))

    # exit(0)

    print(datetime.datetime.now())
    # print("Reading fragments in %s" % total_roi)
    # fragments = fragments[total_roi]

    # segmentation_data = fragments.to_ndarray()

    out_dataset_base = out_dataset

    # create a segmentation
    for threshold in thresholds:
        print(datetime.datetime.now())
        print("Merging for threshold %f..." % threshold)
        # start = time.time()

        # rag.get_segmentation(threshold, segmentation_data)
        # components = rag.get_connected_components(threshold)
        components = get_connected_components(rag, threshold)

        print("Constructing dictionary from fragments to segments")
        fragments_map = {
            fragment: component[0]
            for component in components
            for fragment in component}

        # print(datetime.datetime.now())
        # store segmentation
        print("Writing segmentation...")

        out_dataset = out_dataset_base + "_%.3f" % threshold

        # segmentation = daisy.prepare_ds(
        #     out_file,
        #     out_dataset,
        #     fragments.roi,
        #     fragments.voxel_size,
        #     fragments.data.dtype,
        #     # temporary fix until
        #     # https://github.com/zarr-developers/numcodecs/pull/87 gets approved
        #     # (we want gzip to be the default)
        #     compressor={'id': 'zlib', 'level':5})
        # segmentation.data[:] = segmentation_data
        parallel_relabel(
            fragments_map,
            fragments_file,
            fragments_dataset,
            total_roi,
            block_size=(4080, 4096, 4096),
            seg_file=out_file,
            seg_dataset=out_dataset,
            num_workers=num_workers,
            retry=0)

    print(datetime.datetime.now())


if __name__ == "__main__":

    # configs = {}
    # for config in sys.argv[1:]:
    #     with open(config, 'r') as f:
    #         configs = {**json.load(f), **configs}
    # task_helper.aggregateConfigs(configs)

    user_configs, global_config = task_helper.parseConfigs(sys.argv[1:])

    print(user_configs)
    print(global_config["SegmentationTask"])

    extract_segmentation(**user_configs, **global_config["SegmentationTask"])

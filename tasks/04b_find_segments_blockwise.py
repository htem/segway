import daisy
import json
import logging
import sys
import time
import os

import pymongo
import numpy as np

import task_helper

logging.basicConfig(level=logging.INFO)
logging.getLogger('daisy.persistence.shared_graph_provider').setLevel(logging.DEBUG)
# np.set_printoptions(threshold=sys.maxsize, formatter={'all':lambda x: str(x)})

logger = logging.getLogger(__name__)

def replace_fragment_ids(
        db_host,
        db_name,
        fragments_file,
        lut_dir,
        merge_function,
        roi_offset,
        roi_shape,
        total_roi,
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

        merge_function (``string``):

            The name of the MongoDB database collection to use.

        roi_offset (array-like of ``int``):

            The starting point (inclusive) of the ROI. Entries can be ``None``
            to indicate unboundedness.

        roi_shape (array-like of ``int``):

            The shape of the ROI. Entries can be ``None`` to indicate
            unboundedness.

    '''

    start = time.time()

    lut_dir = os.path.join(
        fragments_file,
        lut_dir)

    for threshold in thresholds:

        print("edges_local2frags_%s_%d/%d.npz" % (merge_function, int(threshold*100), block_id))
        edges = 'edges_local2frags_%s_%d/%d.npz' % (merge_function, int(threshold*100), block_id)
        edges = os.path.join(lut_dir, edges)
        edges = np.load(edges)['edges']
        # print(edges)
        # edges_lut = {n: k for n, k in np.dstack(edges[0], edges[1])}
        # exit(0)

        adj_blocks = generate_adjacent_blocks(roi_offset, roi_shape, total_roi)
        # print(adj_blocks); exit(0)
        for block in adj_blocks:
            adj_block_id = block.block_id

            lookup = 'seg_frags2local_%s_%d/%d.npz' % (merge_function, int(threshold*100), adj_block_id)
            lut = os.path.join(
                    lut_dir,
                    lookup)

            logging.info("Reading %s.." % (lookup))

            if not os.path.exists(lut):
                logging.info("Skipping %s.." % lookup)
                continue

            lut = np.load(lut)['fragment_segment_lut']

            # for frag, comp in lut:
            #     if frag in edges

            # print(np.dstack((lut[0], lut[1])))
            # for e in np.dstack((lut[0], lut[1])):
                # print(e)
            frags2seg = {n: k for n, k in np.dstack((lut[0], lut[1]))[0]}

            for i in range(len(edges)):
                if edges[i][0] in frags2seg:
                    if edges[i][0] != frags2seg[edges[i][0]]:
                        # logging.info("%d remapped to %d" % (edges[i][0], frags2seg[edges[i][0]]))
                        edges[i][0] = frags2seg[edges[i][0]]
                if edges[i][1] in frags2seg:
                    if edges[i][1] != frags2seg[edges[i][1]]:
                        # logging.info("%d remapped to %d" % (edges[i][1], frags2seg[edges[i][1]]))
                        edges[i][1] = frags2seg[edges[i][1]]

            # print(edges)
            # exit(0)

        # print(edges)
        if len(edges):
            # np.unique doesn't work on empty arrays
            edges = np.unique(edges, axis=0)
        # print(edges)
        # print("Length of edges: %d" % len(edges))
        # unique_edges = set([tuple((m, n)) for m, n in edges])
        # edges = [[m, n] for m, n in unique_edges]
        # edges = np.array(edges)
        # print(edges)
        # print("Length of edges after uniquify: %d" % len(edges))
        # exit(0)

        final_edges = 'edges_local2local_%s_%d/%d.npz' % (merge_function, int(threshold*100), block_id)
        out_file = os.path.join(lut_dir, final_edges)

        logger.info("Writing %s" % final_edges)
        np.savez_compressed(out_file, edges=edges)


def generate_adjacent_blocks(roi_offset, roi_shape, total_roi):

    blocks = []
    current_block_roi = daisy.Roi(roi_offset, roi_shape)

    total_write_roi = total_roi.grow(-roi_shape, -roi_shape)

    for offset_mult in [
            (-1, 0, 0),
            (+1, 0, 0),
            (0, -1, 0),
            (0, +1, 0),
            (0, 0, -1),
            (0, 0, +1),
            ]:

        shifted_roi = current_block_roi.shift(roi_shape*offset_mult)
        if total_write_roi.intersects(shifted_roi):
            # print(shifted_roi)
            # print(total_roi)
            blocks.append(
                daisy.Block(total_roi, shifted_roi, shifted_roi))

    # print("Current block: %s" % daisy.Block(total_roi, roi, shifted_roi))
    # print("Adjacent blocks:")
    # for b in blocks:
    #     print(b)

    return blocks


if __name__ == "__main__":

    if sys.argv[1] == 'run':

        user_configs, global_config = task_helper.parseConfigs(sys.argv[2:])
        config = user_configs
        config.update(global_config["FindSegmentsBlockwiseTask2"])

        replace_fragment_ids(**config)

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

        total_roi = daisy.Roi(total_roi_offset, total_roi_shape)

        while True:
            block = client_scheduler.acquire_block()
            if block is None:
                break

            roi_offset = block.write_roi.get_offset()
            roi_shape = daisy.Coordinate(tuple(block_size))

            replace_fragment_ids(
                db_host=db_host,
                db_name=db_name,
                fragments_file=fragments_file,
                lut_dir=lut_dir,
                merge_function=merge_function,
                block_id=block.block_id,
                total_roi=total_roi,
                roi_offset=roi_offset,
                roi_shape=roi_shape,
                thresholds=thresholds,
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

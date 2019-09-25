import json
import os
import logging
# import numpy as np
import sys
import daisy
import pymongo
import time

logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":

    print(sys.argv)
    config_file = sys.argv[1]
    with open(config_file, 'r') as f:
        run_config = json.load(f)

    mask_fragments = False
    mask_file = None
    mask_dataset = None
    epsilon_agglomerate = 0
    use_mahotas = False

    for key in run_config:
        globals()['%s' % key] = run_config[key]

    # open RAG DB
    logging.info("Opening RAG DB...")
    rag_provider = daisy.persistence.MongoDbGraphProvider(
        db_name,
        host=db_host,
        mode='r+',
        directed=False,
        position_attribute=['center_z', 'center_y', 'center_x'])
    logging.info("RAG DB opened")

    '''Illaria TODO
        1. Check for different thresholds
            make them daisy.Parameters
            see the plotting script for these parameters and thesholds: https://github.com/htem/segway/tree/master/synapse_evaluation
    '''

    db_client = pymongo.MongoClient(db_host)
    db = db_client[db_name]
    completion_db = db[completion_db_name]

    print("super_fragments_file: ", super_fragments_file)
    print("super_fragments_dataset: ", super_fragments_dataset)
    print("syn_indicator_file: ", syn_indicator_file)
    print("syn_indicator_dataset: ", syn_indicator_dataset)
    print("syn_dir_file: ", syn_dir_file)
    print("syn_dir_dataset: ", syn_dir_dataset)

    print("WORKER: Running with context %s"%os.environ['DAISY_CONTEXT'])
    client_scheduler = daisy.Client()

    while True:
        block = client_scheduler.acquire_block()
        if block is None:
            break

        logging.info("Running synapse extraction for block %s" % block)

        # TODO: run function to extract synapse for this block

        testing = True
        if testing:
            # FOR TESTING PURPOSES, DON'T RETURN THE BLOCK
            # AND JUST QUIT
            time.sleep(1)
            sys.exit(1)

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

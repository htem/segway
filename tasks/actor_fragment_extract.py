import json
import os
import logging
import numpy as np
import sys
import daisy
import lsd
from lsd.parallel_fragments import watershed_in_block
import pymongo

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
    seed_size = 10  # default seed size from Jan

    for key in run_config:
        globals()['%s' % key] = run_config[key]

    logging.info("Reading affs from %s", affs_file)
    affs = daisy.open_ds(affs_file, affs_dataset, mode='r')

    if mask_fragments:
        raise Exception("Not tested")
        logging.info("Reading mask from %s", mask_file)
        mask = daisy.open_ds(mask_file, mask_dataset, mode='r')
    else:
        mask = None

    fragments_out = daisy.open_ds(
        fragments_file, fragments_dataset, 'r+')

    try:
        myelin_ds = daisy.open_ds(
            fragments_file, myelin_dataset)
    except:
        myelin_ds = None

    # open RAG DB
    logging.info("Opening RAG DB...")
    rag_provider = daisy.persistence.MongoDbGraphProvider(
        db_name,
        host=db_host,
        mode='r+',
        directed=False,
        position_attribute=['center_z', 'center_y', 'center_x'])
    logging.info("RAG DB opened")

    assert fragments_out.data.dtype == np.uint64

    # Tri 5/13/19: disable use_mahotas for now, let me know if this needs to be enabled
    assert use_mahotas == False

    db_client = pymongo.MongoClient(db_host)
    db = db_client[db_name]
    completion_db = db[completion_db_name]

    print("WORKER: Running with context %s"%os.environ['DAISY_CONTEXT'])
    client_scheduler = daisy.Client()

    while True:
        block = client_scheduler.acquire_block()
        if block is None:
            break

        logging.info("Running fragment extraction for block %s" % block)

        watershed_in_block(affs,
                           block,
                           rag_provider,
                           fragments_out,
                           fragments_in_xy,
                           epsilon_agglomerate,
                           mask,
                           myelin_ds=myelin_ds,
                           seed_size=seed_size
                           # use_mahotas=use_mahotas,
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
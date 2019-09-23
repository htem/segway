import json
import os
import logging
# import numpy as np
import sys
import daisy
# import lsd
import pymongo

from segmentation_functions import downsample_with_pooling

logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":

    print(sys.argv)
    config_file = sys.argv[1]
    with open(config_file, 'r') as f:
        run_config = json.load(f)

    downsample_xy = 1

    for key in run_config:
        globals()['%s' % key] = run_config[key]

    logging.info("Reading affs from %s", affs_file)
    affs = daisy.open_ds(affs_file, affs_dataset, mode='r')
    affs_out = daisy.open_ds(affs_file, affs_dataset + '_ds', mode='r+')
    myelin_out = daisy.open_ds(affs_file, 'volumes/myelin_ds', mode='r+')

    try:
        myelin_ds = daisy.open_ds(
            affs_file, myelin_dataset)
    except:
        myelin_ds = None

    db_client = pymongo.MongoClient(db_host)
    db = db_client[db_name]
    completion_db = db[completion_db_name]

    print("WORKER: Running with context %s"%os.environ['DAISY_CONTEXT'])
    client_scheduler = daisy.Client()

    while True:
        block = client_scheduler.acquire_block()
        if block is None:
            break

        logging.info("Running downsample for block %s" % block)

        affs_ds = affs[block.write_roi]
        affs_ds.materialize()
        ds_array = downsample_with_pooling(affs_ds, downsample_xy)
        # print(ds_array.roi)
        # print(ds_array.shape)
        # print(ds_array.data)
        affs_out[block.write_roi] = ds_array

        myelin_dss = myelin_ds[block.write_roi]
        myelin_dss.materialize()
        myelin_array = downsample_with_pooling(myelin_dss, downsample_xy)
        myelin_out[block.write_roi] = myelin_array

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

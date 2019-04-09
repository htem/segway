import json
import os
import logging
import numpy as np
import sys

import daisy
import lsd
from lsd.parallel_fragments import watershed_in_block

logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":

    print(sys.argv)
    config_file = sys.argv[1]
    with open(config_file, 'r') as f:
        run_config = json.load(f)

    mask_fragments = False
    mask_file = None
    mask_dataset = None
    fragments_in_xy = True
    epsilon_agglomerate = 0

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

    # prepare fragments dataset
    fragments_out = daisy.prepare_ds(
        fragments_file,
        fragments_dataset,
        affs.roi,
        affs.voxel_size,
        np.uint64,
        daisy.Roi((0, 0, 0), block_size),
        # temporary fix until
        # https://github.com/zarr-developers/numcodecs/pull/87 gets approved
        # (we want gzip to be the default)
        compressor={'id': 'zlib', 'level':5}
        )

    # open RAG DB
    logging.info("Opening RAG DB...")
    rag_provider = lsd.persistence.MongoDbRagProvider(
        db_name,
        host=db_host,
        mode='r+')
    logging.info("RAG DB opened")

    assert fragments_out.data.dtype == np.uint64

    if context is None:
        context = daisy.Coordinate((0,)*affs.roi.dims())
    else:
        context = daisy.Coordinate(context)

    total_roi = affs.roi.grow(context, context)
    read_roi = daisy.Roi((0,)*affs.roi.dims(), block_size).grow(context, context)
    write_roi = daisy.Roi((0,)*affs.roi.dims(), block_size)

    # fragments_in_xy = fragments_in_xy
    # epsilon_agglomerate = epsilon_agglomerate
    fragments_in_xy = True
    assert(fragments_in_xy)

    print("WORKER: Running with context %s"%os.environ['DAISY_CONTEXT'])
    client_scheduler = daisy.Client()

    while True:
        block = client_scheduler.acquire_block()
        if block is None:
            break

        watershed_in_block(affs,
                           block,
                           rag_provider,
                           fragments_out,
                           fragments_in_xy,
                           epsilon_agglomerate,
                           mask,
                           use_mahotas=use_mahotas)

        client_scheduler.release_block(block, ret=0)
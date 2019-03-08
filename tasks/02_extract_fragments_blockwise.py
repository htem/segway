import json
import logging
import lsd
import numpy as np
import os
import sys

sys.path.append("/groups/funke/home/nguyent3/programming/daisy_stable/")
import daisy

logging.basicConfig(level=logging.INFO)
logging.getLogger('lsd.parallel_fragments').setLevel(logging.DEBUG)
# logging.getLogger('lsd.persistence.sqlite_rag_provider').setLevel(logging.DEBUG)

def extract_fragments(
        affs_file,
        affs_dataset,
        fragments_file,
        fragments_dataset,
        block_size,
        context,
        db_host,
        db_name,
        num_workers,
        fragments_in_xy,
        epsilon_agglomerate=0,
        mask_fragments=False,
        mask_file=None,
        mask_dataset=None):
    '''Run agglomeration in parallel blocks. Requires that affinities have been
    predicted before.

    Args:

        affs_file,
        affs_dataset,
        mask_file,
        mask_dataset (``string``):

            Where to find the affinities and mask (optional).

        block_size (``tuple`` of ``int``):

            The size of one block in world units.

        context (``tuple`` of ``int``):

            The context to consider for fragment extraction and agglomeration,
            in world units.

        db_host (``string``):

            Where to find the MongoDB server.

        db_name (``string``):

            The name of the MongoDB database to use.

        num_workers (``int``):

            How many blocks to run in parallel.

        fragments_in_xy (``bool``):

            Extract fragments section-wise.

        mask_fragments (``bool``):

            Whether to mask fragments for a specified region. Requires that the
            original sample dataset contains a dataset ``volumes/labels/mask``.
    '''

    logging.info("Reading affs from %s", affs_file)
    affs = daisy.open_ds(affs_file, affs_dataset, mode='r')

    if mask_fragments:

        logging.info("Reading mask from %s", mask_file)
        mask = daisy.open_ds(mask_file, mask_dataset, mode='r')

    else:

        mask = None

    # prepare fragments dataset
    fragments = daisy.prepare_ds(
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

    # extract fragments in parallel
    lsd.parallel_watershed(
        affs,
        rag_provider,
        block_size,
        context,
        fragments,
        num_workers,
        fragments_in_xy,
        epsilon_agglomerate=epsilon_agglomerate,
        mask=mask)

if __name__ == "__main__":

    # multiprocessing.set_start_method('spawn', force=True) # for compatibility with multithreaded (namely ioloop)
    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    extract_fragments(**config)
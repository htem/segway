import json
import logging
import numpy as np
import os
import sys

import daisy
import lsd
# from lsd.parallel_aff_agglomerate import agglomerate_in_block
from lsd.parallel_fragments import watershed_in_block

from task_helper import *
from task_01_predict_blockwise import PredictTask

logging.getLogger('lsd.parallel_fragments').setLevel(logging.DEBUG)
# logging.getLogger('lsd.persistence.sqlite_rag_provider').setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)

class ExtractFragmentTask(SlurmTask):

    '''
    Parameters:

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
    
    affs_file = daisy.Parameter()
    affs_dataset = daisy.Parameter()
    block_size = daisy.Parameter()
    context = daisy.Parameter()
    db_host = daisy.Parameter()
    db_name = daisy.Parameter()
    num_workers = daisy.Parameter()

    mask_fragments = daisy.Parameter(default=False)
    mask_file = daisy.Parameter(default=None)
    mask_dataset = daisy.Parameter(default=None)

    fragments_file = daisy.Parameter()
    fragments_dataset = daisy.Parameter()
    fragments_in_xy = daisy.Parameter(default=False)

    epsilon_agglomerate = daisy.Parameter(default=0)

    def prepare(self):
        '''Daisy calls `prepare` for each task prior to scheduling
        any block.'''

        logging.info("Reading affs from %s", self.affs_file)
        self.affs = daisy.open_ds(self.affs_file, self.affs_dataset, mode='r')

        if self.mask_fragments:
            logging.info("Reading mask from %s", self.mask_file)
            self.mask = daisy.open_ds(self.mask_file, self.mask_dataset, mode='r')
        else:
            self.mask = None

        # prepare fragments dataset
        self.fragments_out = daisy.prepare_ds(
            self.fragments_file,
            self.fragments_dataset,
            self.affs.roi,
            self.affs.voxel_size,
            np.uint64,
            daisy.Roi((0, 0, 0), self.block_size),
            # temporary fix until
            # https://github.com/zarr-developers/numcodecs/pull/87 gets approved
            # (we want gzip to be the default)
            compressor={'id': 'zlib', 'level':5}
            )

        # open RAG DB
        logging.info("Opening RAG DB...")
        self.rag_provider = lsd.persistence.MongoDbRagProvider(
            self.db_name,
            host=self.db_host,
            mode='r+')
        logging.info("RAG DB opened")

        assert self.fragments_out.data.dtype == np.uint64

        if self.context is None:
            self.context = daisy.Coordinate((0,)*self.affs.roi.dims())
        else:
            self.context = daisy.Coordinate(self.context)

        total_roi = self.affs.roi.grow(self.context, self.context)
        read_roi = daisy.Roi((0,)*self.affs.roi.dims(), self.block_size).grow(self.context, self.context)
        write_roi = daisy.Roi((0,)*self.affs.roi.dims(), self.block_size)

        config = {
            'affs_file': self.affs_file,
            'affs_dataset': self.affs_dataset,
            'mask_file': self.mask_file,
            'mask_dataset': self.mask_dataset,
            'block_size': self.block_size,
            'context': self.context,
            'db_host': self.db_host,
            'db_name': self.db_name,
            'num_workers': self.num_workers,
            'fragments_in_xy': self.fragments_in_xy,
            'mask_fragments': self.mask_fragments,
            'fragments_file': self.fragments_file,
            'fragments_dataset': self.fragments_dataset,
            'epsilon_agglomerate': self.epsilon_agglomerate,
        }

        self.slurmSetup(config, 'actor_fragment_extract.py')

        self.schedule(
            total_roi=total_roi,
            read_roi=read_roi,
            write_roi=write_roi,
            process_function=self.new_actor,
            check_function=(self.check, lambda b: True),
            read_write_conflict=False,
            fit='shrink',
            num_workers=self.num_workers)

    def check(self, block):
        return self.rag_provider.num_nodes(block.write_roi) > 0

    def requires(self):
        return [PredictTask()]


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    configs = {}
    for config in sys.argv[1:]:
        with open(config, 'r') as f:
            configs = {**json.load(f), **configs}
    aggregateConfigs(configs)
    print(configs)

    daisy.distribute([{'task': ExtractFragmentTask(), 'request': None}],
                     global_config=global_config)

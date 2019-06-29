# import json
import logging
import numpy as np
# import os
import sys

import daisy
import lsd

import task_helper
from task_01_predict_blockwise import PredictTask
from task_merge_myelin import MergeMyelinTask

# logging.getLogger('lsd.parallel_fragments').setLevel(logging.DEBUG)
# logging.getLogger('lsd.persistence.sqlite_rag_provider').setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)


class ExtractFragmentTask(task_helper.SlurmTask):

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

    # sub_roi is used to specify the region of interest while still allocating
    # the entire input raw volume. It is useful when there is a chance that
    # sub_roi will be increased in the future.
    sub_roi_offset = daisy.Parameter(None)
    sub_roi_shape = daisy.Parameter(None)

    mask_fragments = daisy.Parameter(default=False)
    mask_file = daisy.Parameter(default=None)
    mask_dataset = daisy.Parameter(default=None)

    fragments_file = daisy.Parameter()
    fragments_dataset = daisy.Parameter()
    fragments_in_xy = daisy.Parameter()

    raw_file = daisy.Parameter(None)
    raw_dataset = daisy.Parameter(None)

    epsilon_agglomerate = daisy.Parameter(default=0)
    use_mahotas = daisy.Parameter()

    use_myelin_net = daisy.Parameter(default=False)

    def prepare(self):
        '''Daisy calls `prepare` for each task prior to scheduling
        any block.'''

        logging.info("Reading affs from %s", self.affs_file)
        self.affs = daisy.open_ds(self.affs_file, self.affs_dataset, mode='r')

        if self.mask_fragments:
            logging.info("Reading mask from %s", self.mask_file)
            self.mask = daisy.open_ds(self.mask_file, self.mask_dataset,
                                      mode='r')
        else:
            self.mask = None

        # open RAG DB
        logging.info("Opening RAG DB...")
        self.rag_provider = lsd.persistence.MongoDbRagProvider(
            self.db_name,
            host=self.db_host,
            mode='r+')
        logging.info("RAG DB opened")

        if self.context is None:
            self.context = daisy.Coordinate((0,)*self.affs.roi.dims())
        else:
            self.context = daisy.Coordinate(self.context)

        if self.fragments_in_xy:
            # for CB2
            # if we extract fragments in xy, there is no need to have context in Z
            self.context = [n for n in self.context]
            self.context[0] = 0
            self.context = tuple(self.context)

        if self.sub_roi_offset is not None and self.sub_roi_shape is not None:

            # get ROI of source
            assert self.raw_file is not None and self.raw_dataset is not None
            source = daisy.open_ds(self.raw_file, self.raw_dataset)

            # prepare fragments dataset
            self.fragments_out = daisy.prepare_ds(
                self.fragments_file,
                self.fragments_dataset,
                source.roi,
                source.voxel_size,
                np.uint64,
                daisy.Roi((0, 0, 0), self.block_size),
                compressor={'id': 'zlib', 'level': 5}
                )

            total_roi = daisy.Roi(
                tuple(self.sub_roi_offset), tuple(self.sub_roi_shape))
            total_roi = total_roi.grow(self.context, self.context)
            read_roi = daisy.Roi((0,)*total_roi.dims(),
                                 self.block_size).grow(self.context, self.context)
            write_roi = daisy.Roi((0,)*total_roi.dims(), self.block_size)

        else:

            # prepare fragments dataset
            self.fragments_out = daisy.prepare_ds(
                self.fragments_file,
                self.fragments_dataset,
                self.affs.roi,
                self.affs.voxel_size,
                np.uint64,
                daisy.Roi((0, 0, 0), self.block_size),
                compressor={'id': 'zlib', 'level': 5}
                )

            total_roi = self.affs.roi.grow(self.context, self.context)
            read_roi = daisy.Roi((0,)*self.affs.roi.dims(),
                                 self.block_size).grow(self.context, self.context)
            write_roi = daisy.Roi((0,)*self.affs.roi.dims(), self.block_size)

        assert self.fragments_out.data.dtype == np.uint64

        print("total_roi: ", total_roi)
        print("read_roi: ", read_roi)
        print("write_roi: ", write_roi)

        config = {
            'affs_file': self.affs_file,
            'affs_dataset': self.affs_dataset,
            'myelin_dataset': 'volumes/myelin',
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
            'use_mahotas': self.use_mahotas
        }

        self.slurmSetup(config, 'actor_fragment_extract.py')

        check_function = (self.check, lambda b: True)
        if self.overwrite:
            check_function = None

        self.schedule(
            total_roi=total_roi,
            read_roi=read_roi,
            write_roi=write_roi,
            process_function=self.new_actor,
            check_function=check_function,
            read_write_conflict=False,
            fit='shrink',
            num_workers=self.num_workers)

    def check(self, block):
        return self.rag_provider.num_nodes(block.write_roi) > 0

    def requires(self):
        if self.no_check_dependency:
            return []
        if self.use_myelin_net:
            return [MergeMyelinTask(global_config=self.global_config)]
        else:
            return [PredictTask(global_config=self.global_config)]


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    user_configs, global_config = task_helper.parseConfigs(sys.argv[1:])

    req_roi = None
    if "request_offset" in global_config["Input"]:
        req_roi = daisy.Roi(
            tuple(global_config["Input"]["request_offset"]),
            tuple(global_config["Input"]["request_shape"]))
        req_roi = [req_roi]

    daisy.distribute(
        [{'task': ExtractFragmentTask(global_config=global_config,
                                      **user_configs),
         'request': req_roi}],
        global_config=global_config)

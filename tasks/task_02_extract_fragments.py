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

    overwrite_mask_f = daisy.Parameter(None)
    overwrite_sections = daisy.Parameter(None)

    seed_size = daisy.Parameter(10)  # default seed size from Jan

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

        delete_ds = False
        if self.overwrite:
            delete_ds = True

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
                compressor={'id': 'zlib', 'level': 5},
                delete=delete_ds,
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
                compressor={'id': 'zlib', 'level': 5},
                delete=delete_ds,
                )

            total_roi = self.affs.roi.grow(self.context, self.context)
            read_roi = daisy.Roi((0,)*self.affs.roi.dims(),
                                 self.block_size).grow(self.context, self.context)
            write_roi = daisy.Roi((0,)*self.affs.roi.dims(), self.block_size)

        assert self.fragments_out.data.dtype == np.uint64

        self.overwrite_mask = None
        if self.overwrite_mask_f:
            # force precheck = False for any ROI with any voxel in mask = 1
            self.overwrite_mask = daisy.open_ds(
                self.overwrite_mask_f, "overwrite_mask")

        if self.overwrite_sections is not None:
            write_shape = [k for k in total_roi.get_shape()]
            write_shape[0] = 40
            write_shape = tuple(write_shape)

            rois = []
            for s in self.overwrite_sections:
                write_offset = [k for k in total_roi.get_begin()]
                write_offset[0] = s*40
                rois.append(daisy.Roi(write_offset, write_shape))

            self.overwrite_sections = rois

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
            'use_mahotas': self.use_mahotas,
            'seed_size': self.seed_size
        }

        self.slurmSetup(config, 'actor_fragment_extract.py')

        # check_function = (self.check, lambda b: True)
        # if self.overwrite:
        #     check_function = None
        check_function = (
                lambda b: self.check(b, precheck=True),
                lambda b: self.check(b, precheck=False)
                )
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

    def check(self, block, precheck):

        # write_roi = block.write_roi

        if precheck and self.overwrite_sections is not None:
            read_roi_mask = self.overwrite_mask.roi.intersect(block.read_roi)
            for roi in self.overwrite_sections:
                if roi.intersects(read_roi_mask):
                    logger.debug("Block overlaps overwrite_sections %s" % roi)
                    return False

        if precheck and self.overwrite_mask:
            read_roi_mask = self.overwrite_mask.roi.intersect(block.read_roi)
            if not read_roi_mask.empty():
                try:
                    sum = np.sum(self.overwrite_mask[read_roi_mask].to_ndarray())
                    if sum != 0:
                        logger.debug("Block inside overwrite_mask")
                        return False
                except:
                    return False

        if self.completion_db.count({'block_id': block.block_id}) >= 1:
            logger.debug("Skipping block with db check")
            return True

        # check using rag_provider.num_nodes for compatibility with older runs
        done = self.rag_provider.num_nodes(block.write_roi) > 0
        if done:
            self.recording_block_done(block)
            return True

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

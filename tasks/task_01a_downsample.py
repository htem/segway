# import json
import logging
# import numpy as np
# import os
import sys

import daisy
# import lsd

import task_helper2 as task_helper
from task_01_predict_blockwise import PredictTask
# from task_merge_myelin import MergeMyelinTask

# logging.getLogger('lsd.parallel_fragments').setLevel(logging.DEBUG)
# logging.getLogger('lsd.persistence.sqlite_rag_provider').setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)


class DownsampleTask(task_helper.SlurmTask):

    affs_file = daisy.Parameter()
    affs_dataset = daisy.Parameter()
    block_size = daisy.Parameter()
    db_host = daisy.Parameter()
    db_name = daisy.Parameter()
    num_workers = daisy.Parameter()

    # sub_roi is used to specify the region of interest while still allocating
    # the entire input raw volume. It is useful when there is a chance that
    # sub_roi will be increased in the future.
    sub_roi_offset = daisy.Parameter(None)
    sub_roi_shape = daisy.Parameter(None)

    downsample_xy = daisy.Parameter(1)
    context = daisy.Parameter((0, 0, 0))

    def prepare(self):
        '''Daisy calls `prepare` for each task prior to scheduling
        any block.'''

        logging.info("Reading affs from %s", self.affs_file)
        self.affs = daisy.open_ds(self.affs_file, self.affs_dataset, mode='r')

        delete_ds = False
        if self.overwrite:
            delete_ds = True

        voxel_ds_factors = (1, self.downsample_xy, self.downsample_xy)

        self.affs_ds_out = daisy.prepare_ds(
            self.affs_file,
            self.affs_dataset + '_ds',
            self.affs.roi,
            self.affs.voxel_size*voxel_ds_factors,
            self.affs.dtype,
            daisy.Roi((0, 0, 0), self.block_size),
            compressor={'id': 'zlib', 'level': 5},
            delete=delete_ds,
            num_channels=3
            )

        self.myelin_ds_out = daisy.prepare_ds(
            self.affs_file,
            'volumes/myelin' + '_ds',
            self.affs.roi,
            self.affs.voxel_size*voxel_ds_factors,
            self.affs.dtype,
            daisy.Roi((0, 0, 0), self.block_size),
            compressor={'id': 'zlib', 'level': 5},
            delete=delete_ds,
            )

        if self.sub_roi_offset is not None and self.sub_roi_shape is not None:

            total_roi = daisy.Roi(
                tuple(self.sub_roi_offset), tuple(self.sub_roi_shape))
            total_roi = total_roi.grow(self.context, self.context)
            read_roi = daisy.Roi((0,)*total_roi.dims(),
                                 self.block_size).grow(self.context, self.context)
            write_roi = daisy.Roi((0,)*total_roi.dims(), self.block_size)

        else:

            total_roi = self.affs.roi.grow(self.context, self.context)
            read_roi = daisy.Roi((0,)*self.affs.roi.dims(),
                                 self.block_size).grow(self.context, self.context)
            write_roi = daisy.Roi((0,)*self.affs.roi.dims(), self.block_size)

        config = {
            'affs_file': self.affs_file,
            'affs_dataset': self.affs_dataset,
            'myelin_dataset': 'volumes/myelin',
            # 'mask_file': self.mask_file,
            # 'mask_dataset': self.mask_dataset,
            'block_size': self.block_size,
            # 'context': self.context,
            'db_host': self.db_host,
            'db_name': self.db_name,
            'num_workers': self.num_workers,
            # 'fragments_in_xy': self.fragments_in_xy,
            # 'mask_fragments': self.mask_fragments,
            # 'fragments_file': self.fragments_file,
            # 'fragments_dataset': self.fragments_dataset,
            # 'epsilon_agglomerate': self.epsilon_agglomerate,
            'downsample_xy': self.downsample_xy,
            # 'use_mahotas': self.use_mahotas
        }

        self.slurmSetup(config, 'actor_downsample_by_pooling.py')

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

        if self.completion_db.count({'block_id': block.block_id}) >= 1:
            logger.debug("Skipping block with db check")
            return True

        return False

    def requires(self):
        if self.no_check_dependency:
            return []
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
        [{'task': DownsampleTask(global_config=global_config,
                                      **user_configs),
         'request': req_roi}],
        global_config=global_config)

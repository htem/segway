# import json
import logging
import sys
import os
import os.path as path

import numpy as np

import daisy

import task_helper
from task_03_agglomerate_blockwise import AgglomerateTask

logger = logging.getLogger(__name__)


class FindSegmentsBlockwiseTask(task_helper.SlurmTask):

    fragments_file = daisy.Parameter()
    fragments_dataset = daisy.Parameter()
    # out_file = daisy.Parameter()
    # out_dataset = daisy.Parameter()
    merge_function = daisy.Parameter()
    edges_collection = daisy.Parameter()
    thresholds = daisy.Parameter()
    num_workers = daisy.Parameter()
    lut_dir = daisy.Parameter()

    sub_roi_offset = daisy.Parameter(None)
    sub_roi_shape = daisy.Parameter(None)

    block_size = daisy.Parameter()

    def prepare(self):
        '''Daisy calls `prepare` for each task prior to scheduling
        any block.'''

        fragments = daisy.open_ds(self.fragments_file, self.fragments_dataset)

        if self.sub_roi_offset is not None and self.sub_roi_shape is not None:
            total_roi = daisy.Roi(
                tuple(self.sub_roi_offset), tuple(self.sub_roi_shape))
        else:
            total_roi = fragments.roi

        assert fragments.roi.contains(total_roi)

        read_roi = daisy.Roi((0,)*total_roi.dims(), self.block_size)
        write_roi = read_roi

        self.out_dir = os.path.join(
            self.fragments_file,
            self.lut_dir)

        self.last_threshold = self.thresholds[-1]

        for threshold in self.thresholds:
            os.makedirs(os.path.join(self.out_dir, "edges_local2local_%s_%d" % (self.merge_function, int(threshold*100))), exist_ok=True)
            os.makedirs(os.path.join(self.out_dir, "edges_local2frags_%s_%d" % (self.merge_function, int(threshold*100))), exist_ok=True)
            os.makedirs(os.path.join(self.out_dir, "nodes_%s_%d" % (self.merge_function, int(threshold*100))), exist_ok=True)
            os.makedirs(os.path.join(self.out_dir, "seg_frags2local_%s_%d" % (self.merge_function, int(threshold*100))), exist_ok=True)

        config = {
            'db_host': self.db_host,
            'db_name': self.db_name,
            'fragments_file': self.fragments_file,
            'lut_dir': self.lut_dir,
            'merge_function': self.merge_function,
            'edges_collection': self.edges_collection,
            # 'num_workers': self.num_workers,
            'thresholds': self.thresholds,
        }
        self.slurmSetup(
            config,
            '04a_find_segments_blockwise.py')

        check_function = self.block_done
        if self.overwrite:
            check_function = None

        self.schedule(
            total_roi,
            read_roi,
            write_roi,
            process_function=self.new_actor,
            check_function=check_function,
            num_workers=self.num_workers,
            read_write_conflict=False,
            fit='shrink')

    def requires(self):
        if self.no_check_dependency:
            return []
        # return [AgglomerateTask(global_config=self.global_config)]
        return []

    def block_done(self, block):

        block_id = block.block_id
        lookup = 'edges_local2frags_%s_%d/%d.npz' % (
            self.merge_function, int(self.last_threshold*100), block_id)
        out_file = os.path.join(self.out_dir, lookup)
        logger.debug("Checking %s" % out_file)
        exists = path.exists(out_file)
        return exists


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    user_configs, global_config = task_helper.parseConfigs(sys.argv[1:])

    daisy.distribute(
        [{'task': FindSegmentsBlockwiseTask(global_config=global_config,
                                            **user_configs),
         'request': None}],
        global_config=global_config)

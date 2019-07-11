# import json
import logging
import sys

import daisy

import task_helper
from task_03_agglomerate_blockwise import AgglomerateTask

logger = logging.getLogger(__name__)


class SegmentationTask(task_helper.SlurmTask):

    fragments_file = daisy.Parameter()
    fragments_dataset = daisy.Parameter()
    out_file = daisy.Parameter()
    out_dataset = daisy.Parameter()
    db_host = daisy.Parameter()
    db_name = daisy.Parameter()
    edges_collection = daisy.Parameter()
    thresholds = daisy.Parameter()
    num_workers = daisy.Parameter(default=4)

    def prepare(self):
        '''Daisy calls `prepare` for each task prior to scheduling
        any block.'''

        logger.info("Opening fragments {}".format(self.fragments_file))
        fragments = daisy.open_ds(self.fragments_file, self.fragments_dataset)
        total_roi = fragments.roi

        config = {
            'fragments_file': self.fragments_file,
            'fragments_dataset': self.fragments_dataset,
            'out_file': self.out_file,
            'out_dataset': self.out_dataset,
            'db_host': self.db_host,
            'db_name': self.db_name,
            'edges_collection': self.edges_collection,
            'thresholds': self.thresholds,
            # 'roi_offset': self.roi_offset,
            # 'roi_shape': self.roi_shape,
            'num_workers': self.num_workers
        }
        print(config)
        self.slurmSetup(
            config,
            'actor_segmentation.py')

        check_function = self.block_done
        if self.overwrite:
            check_function = None

        self.schedule(
            total_roi,
            total_roi,
            total_roi,
            process_function=self.new_actor,
            check_function=check_function,
            num_workers=1,
            read_write_conflict=False,
            fit='shrink')

    def requires(self):
        if self.no_check_dependency or (not self.overwrite and self.is_written()):
            return []
        return [AgglomerateTask(global_config=self.global_config)]

    def block_done(self, block):
        return self.is_written()

    def is_written(self):
        # check if one of the segment dataset is written
        out_dataset = self.out_dataset + "_%.3f" % self.thresholds[-1]
        try:
            daisy.open_ds(self.out_file, out_dataset)
        except:
            return False
        return True


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    user_configs, global_config = task_helper.parseConfigs(sys.argv[1:])

    daisy.distribute(
        [{'task': SegmentationTask(global_config=global_config,
                                   **user_configs),
         'request': None}],
        global_config=global_config)


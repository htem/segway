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
    # roi_offset = daisy.Parameter(default=None)
    # roi_shape = daisy.Parameter()
    num_workers = daisy.Parameter(default=4)
    no_check = daisy.Parameter(default=0)

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
        self.slurmSetup(
            config,
            'actor_segmentation.py')

        self.schedule(
            total_roi,
            total_roi,
            total_roi,
            process_function=self.new_actor,
            num_workers=1,
            read_write_conflict=False,
            fit='shrink')

    def requires(self):
        if self.no_check:
            return []
        return [AgglomerateTask(global_config=self.global_config)]


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    user_configs, global_config = task_helper.parseConfigs(sys.argv[1:])

    daisy.distribute(
        [{'task': SegmentationTask(global_config=global_config,
                                   **user_configs),
         'request': None}],
        global_config=global_config)

    # configs = {}
    # user_configs = {}
    # for config in sys.argv[1:]:
    #     if "=" in config:
    #         key, val = config.split('=')
    #         user_configs[key] = val
    #     else:
    #         with open(config, 'r') as f:
    #             configs = {**json.load(f), **configs}
    # task_helper.aggregateConfigs(configs)
    # print(configs)

    # daisy.distribute([{'task': SegmentationTask(**user_configs), 'request': None}],
    #                  global_config=configs)

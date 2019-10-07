# import json
import logging
# import numpy as np
# import os
import sys

import daisy

from segway.tasks import task_helper2 as task_helper
from task_01_predict_blockwise import PredictSynapseTask

# logging.getLogger('lsd.parallel_fragments').setLevel(logging.DEBUG)
# logging.getLogger('lsd.persistence.sqlite_rag_provider').setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)


class ExtractSynapsesTask(task_helper.SlurmTask):

    '''
    README
    '''

    super_fragments_file = daisy.Parameter()
    super_fragments_dataset = daisy.Parameter()

    syn_indicator_file = daisy.Parameter()
    syn_indicator_dataset = daisy.Parameter()

    syn_dir_file = daisy.Parameter()
    syn_dir_dataset = daisy.Parameter()

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

    def prepare(self):
        '''Daisy calls `prepare` for each task prior to scheduling
        any block.'''

        # open RAG DB
        logging.info("Opening RAG DB...")
        self.rag_provider = daisy.persistence.MongoDbGraphProvider(
            self.db_name,
            host=self.db_host,
            mode='r+',
            directed=False,
            # edges_collection='edges_' + self.merge_function,
            # position_attribute=['center_z', 'center_y', 'center_x'],
            # TODO: I'm not sure if you'd need to specify the edges and position attributes
            # here
            )
        logging.info("RAG DB opened")

        if self.context is None:
            self.context = daisy.Coordinate((0,)*self.affs.roi.dims())
        else:
            self.context = daisy.Coordinate(self.context)

        if self.sub_roi_offset is not None and self.sub_roi_shape is not None:

            assert False, "Untested"  # I will make this right later

            # get ROI of source
            assert self.raw_file is not None and self.raw_dataset is not None
            # source = daisy.open_ds(self.raw_file, self.raw_dataset)

            total_roi = daisy.Roi(
                tuple(self.sub_roi_offset), tuple(self.sub_roi_shape))
            total_roi = total_roi.grow(self.context, self.context)
            read_roi = daisy.Roi((0,)*total_roi.dims(),
                                 self.block_size).grow(self.context, self.context)
            write_roi = daisy.Roi((0,)*total_roi.dims(), self.block_size)

        else:
            # get ROIs from an existing dataset
            syn_indicator_ds = daisy.open_ds(
                self.syn_indicator_file, self.syn_indicator_dataset, 'r')

            total_roi = syn_indicator_ds.roi.grow(self.context, self.context)
            read_roi = daisy.Roi((0,)*syn_indicator_ds.roi.dims(),
                                 self.block_size).grow(self.context, self.context)
            write_roi = daisy.Roi((0,)*syn_indicator_ds.roi.dims(), self.block_size)

        '''Illaria TODO
            1. Check for different thresholds
                make them daisy.Parameters
                see the plotting script for these parameters and thesholds: https://github.com/htem/segway/tree/master/synapse_evaluation
        '''

        # parameters needed to run the actor python script
        config = {
            'block_size': self.block_size,
            'context': self.context,
            'db_host': self.db_host,
            'db_name': self.db_name,
            'num_workers': self.num_workers,
            'super_fragments_file': self.super_fragments_file,
            'super_fragments_dataset': self.super_fragments_dataset,
            'syn_indicator_file': self.syn_indicator_file,
            'syn_indicator_dataset': self.syn_indicator_dataset,
            'syn_dir_file': self.syn_dir_file,
            'syn_dir_dataset': self.syn_dir_dataset,
        }

        self.slurmSetup(config, 'segway/synful_tasks/actor_02_extract_synapses.py')

        check_function = (
                lambda b: self.check(b, precheck=True),
                lambda b: self.check(b, precheck=False)
                )
        if self.overwrite:
            # do not check (check_fn = null) if `overwrite` is True
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
        '''By default only check the `completion_db` database that gets
        marked for each finished block by the worker.
        '''

        if self.completion_db.count({'block_id': block.block_id}) >= 1:
            logger.debug("Skipping block with db check")
            return True

        return False

    def requires(self):
        if self.no_check_dependency:
            return []
        else:
            return [PredictSynapseTask(global_config=self.global_config)]


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
        [{'task': ExtractSynapsesTask(global_config=global_config,
                                      **user_configs),
         'request': req_roi}],
        global_config=global_config)
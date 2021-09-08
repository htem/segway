# import json
import logging
# import numpy as np
# import os
import sys

import daisy

from segway.tasks import task_helper2 as task_helper
from task_01_predict_blockwise import PredictSynapseTask
from database_synapses import SynapseDatabase
from database_superfragments import SuperFragmentDatabase

# logging.getLogger('lsd.parallel_fragments').setLevel(logging.DEBUG)
# logging.getLogger('lsd.persistence.sqlite_rag_provider').setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)


class ExtractSynapsesTask(task_helper.SlurmTask):

    '''
    README
    '''

    super_fragments_file = daisy.Parameter()
    super_fragments_dataset = daisy.Parameter()

    raw_file = daisy.Parameter(None)
    raw_dataset = daisy.Parameter(None)
    affs_file = daisy.Parameter(None)
    affs_dataset = daisy.Parameter(None)

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

    score_threshold = daisy.Parameter(0.7)

    # cc or nms
    extract_type = daisy.Parameter("cc")
    cc_threshold = daisy.Parameter(0.25)
    loc_type = daisy.Parameter('edt')
    score_type = daisy.Parameter('mean')
    db_col_name_syn = daisy.Parameter('synapses')
    db_col_name_sf = daisy.Parameter('superfragments')
    # nms_radius = daisy.Parameter()

    prediction_mode = daisy.Parameter("post_to_pre")
    remove_z_dir = daisy.Parameter(False)
    d_vector_scale = daisy.Parameter(None)

    realignment_xy_context_nm = daisy.Parameter(1024)
    realignment_xy_stride_nm = daisy.Parameter(1024)

    def prepare(self):
        '''Daisy calls `prepare` for each task prior to scheduling
        any block.'''

        mode = 'r+'
        if self.overwrite:
            mode = 'w'

        syn_db = SynapseDatabase(self.db_name, self.db_host, self.db_col_name_syn, mode=mode)
        sf_db = SuperFragmentDatabase(self.db_name, self.db_host, self.db_col_name_sf, mode=mode)

        # print("self.db_name: ", self.db_name)
        # print("self.db_host: ", self.db_host)
        # print("self.db_col_name_syn: ", self.db_col_name_syn)
        # print("mode: ", mode)

        affs_ds = None
        if self.affs_file or self.affs_dataset:
            assert self.affs_file and self.affs_dataset
            affs_ds = daisy.open_ds(self.affs_file, self.affs_dataset, 'r') 

        raw_ds = None
        if self.raw_file or self.raw_dataset:
            assert self.raw_file and self.raw_dataset
            raw_ds = daisy.open_ds(self.raw_file, self.raw_dataset, 'r') 

        if self.context is None:
            self.context = daisy.Coordinate((0,)*self.affs.roi.dims())
        else:
            self.context = daisy.Coordinate(self.context)

        if self.sub_roi_offset is not None and self.sub_roi_shape is not None:

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

            roi = syn_indicator_ds.roi
            if affs_ds:
                roi = roi.intersect(affs_ds.roi)
            if raw_ds:
                roi = roi.intersect(raw_ds.roi)

            total_roi = roi.grow(self.context, self.context)
            read_roi = daisy.Roi((0,)*roi.dims(),
                                 self.block_size).grow(self.context, self.context)
            write_roi = daisy.Roi((0,)*roi.dims(), self.block_size)

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
            'affs_file': self.affs_file,
            'affs_dataset': self.affs_dataset,
            'raw_file': self.raw_file,
            'raw_dataset': self.raw_dataset,
            'realignment_xy_context_nm': self.realignment_xy_context_nm,
            'realignment_xy_stride_nm': self.realignment_xy_stride_nm,

            'score_threshold': self.score_threshold,
            'extract_type' : self.extract_type,
            'cc_threshold' : self.cc_threshold,
            'loc_type' : self.loc_type,
            'score_type' : self.score_type,
            'db_col_name_syn' : self.db_col_name_syn,
            'db_col_name_sf': self.db_col_name_sf,
            'prediction_mode': self.prediction_mode,
            'remove_z_dir': self.remove_z_dir,
            'd_vector_scale': self.d_vector_scale,
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
            num_workers=self.num_workers,
            max_retries=self.max_retries,
            )

    def check(self, block, precheck):
        '''By default only check the `completion_db` database that gets
        marked for each finished block by the worker.
        '''
        if self.completion_db.count({'block_id': block.block_id}) >= 1:
            logger.info("Skipping block with db check")
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

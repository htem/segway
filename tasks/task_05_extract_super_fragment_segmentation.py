import logging
# import lsd
# import numpy as np
import daisy
# from daisy import Coordinate, Roi
import sys
import numpy as np

import os

import task_helper2 as task_helper
from task_04bb_find_segment_blockwise import FindSegmentsBlockwiseTask2b

logger = logging.getLogger(__name__)


class ExtractSuperFragmentSegmentationTask(task_helper.SlurmTask):
    '''Segment .

    These seeds are assumed to belong to the same segment.
    '''

    fragments_file = daisy.Parameter()
    fragments_dataset = daisy.Parameter()
    out_dataset = daisy.Parameter()
    out_file = daisy.Parameter()
    context = daisy.Parameter([0, 0, 0])
    db_host = daisy.Parameter()
    db_name = daisy.Parameter()
    thresholds = daisy.Parameter()
    merge_function = daisy.Parameter()
    num_workers = daisy.Parameter(1)
    lut_dir = daisy.Parameter()

    sub_roi_offset = daisy.Parameter(None)
    sub_roi_shape = daisy.Parameter(None)

    super_chunk_size = daisy.Parameter()
    block_size = daisy.Parameter()
    write_size = daisy.Parameter()

    ds_roi_offset = daisy.Parameter(None)

    def prepare(self):
        '''Daisy calls `prepare` for each task prior to scheduling
        any block.'''

        self.block_size = (
            daisy.Coordinate(self.block_size) * tuple(self.super_chunk_size))

        logging.info("Reading fragments from %s", self.fragments_file)
        self.fragments = daisy.open_ds(self.fragments_file,
                                       self.fragments_dataset,
                                       mode='r')

        if self.sub_roi_offset is not None and self.sub_roi_shape is not None:
            total_roi = daisy.Roi(
                tuple(self.sub_roi_offset), tuple(self.sub_roi_shape))
        else:
            total_roi = self.fragments.roi

        read_roi = daisy.Roi((0,)*self.fragments.roi.dims(), self.block_size)
        write_roi = read_roi

        ds_roi = self.fragments.roi
        print("original ds_roi", ds_roi)
        if self.ds_roi_offset is not None:
            self.ds_roi_offset = daisy.Coordinate(self.ds_roi_offset)
            ds_roi = ds_roi.grow(-self.ds_roi_offset, (0, 0, 0))
            print("ds_roi", ds_roi)

        for threshold in self.thresholds:
            ds = self.out_dataset + "_%.3f" % threshold
            self.segment_ds = daisy.prepare_ds(
                self.out_file,
                ds,
                ds_roi,
                self.fragments.voxel_size,
                self.fragments.data.dtype,
                write_size=daisy.Coordinate(tuple(self.write_size)),
                force_exact_write_size=True,
                compressor={'id': 'zlib', 'level': 5})

        last_threshold = self.thresholds[-1]

        lut_dir = os.path.join(self.fragments_file, self.lut_dir)
        super_lut_dir = 'super_%dx%dx%d_%s' % (
            self.super_chunk_size[0], self.super_chunk_size[1], self.super_chunk_size[2],
            self.merge_function)
        super_lut_dir = os.path.join(lut_dir, super_lut_dir)

        config = {
            'fragments_file': self.fragments_file,
            'fragments_dataset': self.fragments_dataset,
            'lut_dir': lut_dir,
            'super_lut_dir': super_lut_dir,
            'merge_function': self.merge_function,
            'thresholds': self.thresholds,
            'out_dataset': self.out_dataset,
            'out_file': self.out_file,
            'super_chunk_size': self.super_chunk_size,
            'block_size': self.block_size,
            'total_roi_offset': total_roi.get_offset(),
            'total_roi_shape': total_roi.get_shape(),
        }

        self.completion_db_class_name = "Task05"
        self.slurmSetup(
            config,
            '05_super_fragment_segmentation.py',
            completion_db_name_extra="%d" % int(last_threshold*1000))

        check_function = self.check_block
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
            max_retries=0,
            fit='shrink')  # TODO: consider if other strategies valid

    def check_block(self, block):

        logger.debug("Checking if block %s is complete..." % block.write_roi)

        if self.segment_ds.roi.intersect(block.write_roi).empty():
            logger.debug("Block outside of output ROI")
            return True

        if self.completion_db.count({'block_id': block.block_id}) >= 1:
            logger.debug("Skipping block with db check")
            return True

        center_coord = (block.write_roi.get_begin() +
                        block.write_roi.get_end()) / 2
        center_values = self.segment_ds[center_coord]
        s = np.sum(center_values)

        logger.debug("Sum of center values in %s is %f" % (block.write_roi, s))

        done = s != 0
        if done:
            self.recording_block_done(block)

        # TODO: this should be filtered by post check and not pre check
        # if (s == 0):
        #     self.log_error_block(block)

        return done

    def requires(self):
        if self.no_check_dependency:
            return []
        return [FindSegmentsBlockwiseTask2b(global_config=self.global_config)]


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    user_configs, global_config = task_helper.parseConfigs(sys.argv[1:])

    print(global_config)

    daisy.distribute(
        [{'task': ExtractSuperFragmentSegmentationTask(global_config=global_config,
                                             **user_configs),
         'request': None}],
        global_config=global_config)
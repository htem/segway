import json
import logging
import numpy as np
import os
import sys

import daisy

# from task_helper import *
import task_helper
# logging.getLogger('daisy.blocks').setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)


class PredictTask(task_helper.SlurmTask):
    '''Run prediction in parallel blocks. Within blocks, predict in chunks.

    Parameters:

        experiment (``string``):

            Name of the experiment (cremi, fib19, fib25, ...).

        setup (``string``):

            Name of the setup to predict.

        iteration (``int``):

            Training iteration to predict from.

        raw_file (``string``):
        raw_dataset (``string``):
        lsds_file (``string``):
        lsds_dataset (``string``):

            Paths to the input datasets. lsds can be None if not needed.

        out_file (``string``):
        out_dataset (``string``):

            Path to the output datset.

        block_size_in_chunks (``tuple`` of ``int``):

            The size of one block in chunks (not voxels!). A chunk corresponds
            to the output size of the network.

        num_workers (``int``):

            How many blocks to run in parallel.
    '''

    # experiment = daisy.Parameter()
    # setup = daisy.Parameter()
    train_dir = daisy.Parameter()
    iteration = daisy.Parameter()
    raw_file = daisy.Parameter()
    raw_dataset = daisy.Parameter()
    # lsds_file = daisy.Parameter()
    # lsds_dataset = daisy.Parameter()
    out_file = daisy.Parameter()
    out_dataset = daisy.Parameter()
    block_size_in_chunks = daisy.Parameter()
    num_workers = daisy.Parameter()
    predict_file = daisy.Parameter(None)

    output_shape = daisy.Parameter(None)
    out_dtype = daisy.Parameter(None)
    out_dims = daisy.Parameter(None)
    net_voxel_size = daisy.Parameter(None)
    input_shape = daisy.Parameter(None)

    log_to_stdout = daisy.Parameter(default=True)
    log_to_files = daisy.Parameter(default=False)

    cpu_cores = daisy.Parameter(4)

    # predict_num_core = daisy.Parameter(2)

    def prepare(self):
        '''Daisy calls `prepare` for each task prior to scheduling
        any block.'''

        self.setup = os.path.abspath(self.train_dir)
        self.raw_file = os.path.abspath(self.raw_file)
        self.out_file = os.path.abspath(self.out_file)

        logger.info('Input file path: ' + self.raw_file)
        logger.info('Output file path: ' + self.out_file)
        # from here on, all values are in world units (unless explicitly mentioned)

        # get ROI of source
        try:
            source = daisy.open_ds(self.raw_file, self.raw_dataset)
        except Exception:
            raise Exception("Raw dataset not found! "
                            "Please fix file path... "
                            "raw_file: {}".format(self.raw_file))
            # in_dataset = in_dataset + '/s0'
            # source = daisy.open_ds(in_file, in_dataset)

        logger.info("Source dataset has shape %s, ROI %s, voxel size %s"%(
            source.shape, source.roi, source.voxel_size))

        # load config
        # with open(os.path.join(self.setup, 'config.json')) as f:
        #     logger.info("Reading setup config from %s"%os.path.join(self.setup, 'config.json'))
        #     net_config = json.load(f)

        # out_dims = net_config['out_dims']
        # out_dtype = net_config['out_dtype']
        logger.info('Number of dimensions is %i' % self.out_dims)

        # get chunk size and context
        voxel_size = source.voxel_size
        self.net_voxel_size = tuple(self.net_voxel_size)
        # net_voxel_size = daisy.Coordinate(self.voxel_size)
        # print(voxel_size)
        # print(self.net_voxel_size)
        if self.net_voxel_size != source.voxel_size:
            logger.info("Mismatched net and source voxel size. "
                        "Assuming downsampling")
            # force same voxel size for net in and output dataset
            voxel_size = self.net_voxel_size

        net_input_size = daisy.Coordinate(self.input_shape)*voxel_size
        net_output_size = daisy.Coordinate(self.output_shape)*voxel_size
        chunk_size = net_output_size
        context = (net_input_size - net_output_size)/2

        logger.info("Following sizes in world units:")
        logger.info("net input size  = %s" % (net_input_size,))
        logger.info("net output size = %s" % (net_output_size,))
        logger.info("context         = %s" % (context,))
        logger.info("chunk size      = %s" % (chunk_size,))

        # compute sizes of blocks
        block_output_size = chunk_size*tuple(self.block_size_in_chunks)
        block_input_size = block_output_size + context*2

        # get total input and output ROIs
        input_roi = source.roi.grow(context, context)
        output_roi = source.roi

        # create read and write ROI
        block_read_roi = daisy.Roi((0, 0, 0), block_input_size) - context
        block_write_roi = daisy.Roi((0, 0, 0), block_output_size)

        logger.info("Following ROIs in world units:")
        logger.info("Total input ROI  = %s" % input_roi)
        logger.info("Block read  ROI  = %s" % block_read_roi)
        logger.info("Block write ROI  = %s" % block_write_roi)
        logger.info("Total output ROI = %s" % output_roi)

        logging.info('Preparing output dataset')

        ds = daisy.prepare_ds(
            self.out_file,
            self.out_dataset,
            output_roi,
            voxel_size,
            self.out_dtype,
            write_roi=daisy.Roi((0, 0, 0), chunk_size),
            num_channels=self.out_dims,
            # temporary fix until
            # https://github.com/zarr-developers/numcodecs/pull/87 gets approved
            # (we want gzip to be the default)
            compressor={'id': 'zlib', 'level':5}
            )

        if self.raw_file.endswith('.json'):
            with open(self.raw_file, 'r') as f:
                spec = json.load(f)
                self.raw_file = spec['container']

        config = {
            # 'experiment': self.experiment,
            # 'setup': self.setup,
            'iteration': self.iteration,
            'raw_file': self.raw_file,
            'raw_dataset': self.raw_dataset,
            'read_begin': 0,
            'read_size': 0,
            'out_file': self.out_file,
            'out_dataset': self.out_dataset,
            'output_shape': self.output_shape,
            # 'out_dtype': self.out_dtype,
            'voxel_size': self.net_voxel_size,
            'input_shape': self.input_shape,
            'train_dir': self.train_dir,
            'write_begin': 0,
            'write_size': 0,
            'predict_num_core': self.cpu_cores
        }

        if self.predict_file is not None:
            predict_script = self.predict_file
        else:
            # use the one included in folder
            predict_script = '%s/predict.py' % (self.train_dir)
        print(self.predict_file)
        print(predict_script)

        self.cpu_mem = self.cpu_cores*1
        self.slurmSetup(config,
                        predict_script,
                        gpu='any')

        # any task must call schedule() at the end of prepare
        self.schedule(
            total_roi=input_roi,
            read_roi=block_read_roi,
            write_roi=block_write_roi,
            process_function=self.new_actor,
            check_function=(self.check_block, lambda b: True),
            read_write_conflict=False,
            fit='overhang',
            num_workers=self.num_workers,
            # log_to_file=True
            )

    def check_block(self, block):

        logger.debug("Checking if block %s is complete..." % block.write_roi)

        ds = daisy.open_ds(self.out_file, self.out_dataset)
        write_roi = ds.roi.intersect(block.write_roi)
        if write_roi.empty():
            logger.debug("Block outside of output ROI")
            return True

        center_coord = (write_roi.get_begin() +
                        write_roi.get_end()) / 2
        center_values = ds[center_coord]
        s = np.sum(center_values)
        logger.debug("Sum of center values in %s is %f" % (write_roi, s))

        return s != 0


if __name__ == "__main__":
    # logging.basicConfig(level=logging.DEBUG)
    logging.basicConfig(level=logging.INFO)

    user_configs, global_config = task_helper.parseConfigs(sys.argv[1:])

    # 10560:14080, 263680:266752, 341504:344576
    # roi = daisy.Roi(Coordinate([10560, 263680, 341504]),
    #                 Coordinate([3520, 3072, 3072]))
    # roi = daisy.Roi(Coordinate([10560, 263680, 341504]),
    #                 Coordinate([1, 1, 1]))

    daisy.distribute(
        # [{'task': BlockwiseSegmentationTask(**user_configs, request_roi=roi),
        [{'task': PredictTask(global_config=global_config,
                              **user_configs),
         'request': None}],
        global_config=global_config)

    # configs = {}
    # for config in sys.argv[1:]:
    #     with open(config, 'r') as f:
    #         configs = {**json.load(f), **configs}
    # aggregateConfigs(configs)
    # print(configs)

    # daisy.distribute([{'task': PredictTask(), 'request': None}], global_config=configs)

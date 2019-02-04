from __future__ import print_function

import sys

from gunpowder import *
from gunpowder.contrib import ZeroOutConstSections
from gunpowder.tensorflow import *
import json
import logging
# import numpy as np
import os

# sys.path.insert(0, "/groups/funke/home/nguyent3/programming/daisy/")
# import daisy


def predict(
        iteration,
        raw_file,
        raw_dataset,
        # read_roi,
        input_shape,
        output_shape,
        voxel_size,
        out_file,
        out_dataset,
        train_dir,
        predict_num_core):

    # setup_dir = os.path.dirname(os.path.realpath(__file__))
    setup_dir = train_dir

    with open(os.path.join(setup_dir, 'net_io_names.json'), 'r') as f:
        net_config = json.load(f)

    # voxels
    input_shape = Coordinate(input_shape)
    output_shape = Coordinate(output_shape)
    voxel_size = Coordinate(tuple(voxel_size))

    context = (input_shape - output_shape)//2
    # voxel_size = Coordinate(tuple(cfg.get("voxel_size",(40,4,4))))
    # input_size = Coordinate(tuple(cfg.get("input_size_pixels",(84,268,268))))*voxel_size
    # output_size = Coordinate(tuple(cfg.get("output_size_pixels",(56,56,56))))*voxel_size

    print("Context is %s"%(context,))
    # nm
    # voxel_size = Coordinate((40, 4, 4))
    # voxel_size = Coordinate(net_config["voxel_size"])
    # context_nm = context*voxel_size
    input_size = input_shape*voxel_size
    output_size = output_shape*voxel_size

    raw = ArrayKey('RAW')
    affs = ArrayKey('AFFS')

    chunk_request = BatchRequest()
    chunk_request.add(raw, input_size)
    chunk_request.add(affs, output_size)

    mapping = {
        # 'RAW': "read_roi",
        # 'AFFS': "write_roi"
        raw: "read_roi",
        affs: "write_roi"
    }

    pipeline = Hdf5Source(
                   raw_file,
                   datasets = { raw: 'volumes/raw'})

    pipeline += Pad(raw, size=None)

    # pipeline += Crop(raw, read_roi)

    pipeline += Normalize(raw)

    pipeline += IntensityScaleShift(raw, 2,-1)

    # new from logan's
    pipeline += ZeroOutConstSections(raw)

    pipeline += Predict(
            os.path.join(setup_dir, 'unet_checkpoint_%d'%iteration),
            inputs={
                net_config['raw']: raw
            },
            outputs={
                net_config['affs']: affs
            },
            # graph=os.path.join(setup_dir, 'config.meta')
        )

    pipeline += IntensityScaleShift(affs, 255, 0)

    pipeline += ZarrWrite(
            dataset_names={
                affs: out_dataset,
            },
            output_filename=out_file
        )

    pipeline += PrintProfilingStats(every=10)

    pipeline += DaisyScan(chunk_request, mapping, num_workers=predict_num_core)
    # pipeline += DaisyScan(chunk_request, mapping)

    print("Starting prediction...")
    with build(pipeline):
        pipeline.request_batch(BatchRequest())
    print("Prediction finished")

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    # logging.basicConfig(level=logging.DEBUG)
    logging.getLogger('gunpowder.nodes.hdf5like_write_base').setLevel(logging.DEBUG)

    print(sys.argv)
    config_file = sys.argv[1]
    with open(config_file, 'r') as f:
        run_config = json.load(f)

    # read_roi = Roi(
    #     run_config['read_begin'],
    #     run_config['read_size'])
    # write_roi = read_roi.grow(-context_nm, -context_nm)

    # print("Read ROI in nm is %s"%read_roi)
    # print("Write ROI in nm is %s"%write_roi)

    # print(run_config)
        
    predict(
        run_config['iteration'],
        run_config['raw_file'],
        run_config['raw_dataset'],
        # read_roi,
        run_config['input_shape'],
        run_config['output_shape'],
        run_config['voxel_size'],
        run_config['out_file'],
        run_config['out_dataset'],
        run_config['train_dir'],
        run_config['predict_num_core']
        )

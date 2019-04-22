import json
import os
import logging
import numpy as np
import sys
import daisy

from collections import OrderedDict
# from PIL import Image

import ilastik_main
from ilastik.applets.dataSelection import DatasetInfo
from ilastik.workflows.pixelClassification import PixelClassificationWorkflow
from ilastik.workflows import AutocontextTwoStage
import vigra

logging.basicConfig(level=logging.INFO)


def downsample_data(data, factor):
    slices = tuple(
        slice(None, None, k)
        for k in factor)
    return data[slices]


def predict_myelin_2d(
        block,
        raw_ds,
        myelin_ds,
        downsample_xy
        ):

    raw_ndarray = raw_ds[block.read_roi].to_ndarray()
    # we need to slice 3D volume to 2D and downsample it
    downsample_factors = (1, downsample_xy, downsample_xy)
    raw_ndarray = downsample_data(raw_ndarray, downsample_factors)

    inputs = []
    for raw in raw_ndarray:
        input_data = vigra.taggedView(raw, 'yx')
        inputs.append(DatasetInfo(preloaded_array=input_data))

    # Construct an OrderedDict of role-names -> DatasetInfos
    # (See PixelClassificationWorkflow.ROLE_NAMES)
    role_data_dict = OrderedDict(
          [("Raw Data", inputs)])

    # Run the export via the BatchProcessingApplet
    predictions = shell.workflow.batchProcessingApplet.run_export(
        role_data_dict, export_to_array=True)

    # get only myelin labels
    myelin_labels = []
    for o in predictions:
        myelin_labels.append(o[:, :, 0])
    # 2d arrays to 3d array
    out_ndarray = np.stack(myelin_labels)
    # convert float to uint8
    out_ndarray = np.stack(out_ndarray)
    out_ndarray = np.array((1-out_ndarray)*255, dtype=np.uint8)
    myelin_ds[block.write_roi] = out_ndarray


if __name__ == "__main__":

    print(sys.argv)
    config_file = sys.argv[1]
    with open(config_file, 'r') as f:
        run_config = json.load(f)

    # mask_fragments = False
    # mask_file = None
    # mask_dataset = None
    # fragments_in_xy = True
    # epsilon_agglomerate = 0
    raw_file = None
    raw_dataset = None
    myelin_file = None
    myelin_dataset = None
    downsample_xy = None
    lazyflow_num_threads = None
    lazyflow_mem = None
    ilastik_project_path = None

    for key in run_config:
        globals()['%s' % key] = run_config[key]

    logging.info("Reading raw from %s", raw_file)
    raw_ds = daisy.open_ds(raw_file, raw_dataset, mode='r')
    myelin_ds = daisy.open_ds(myelin_file, myelin_dataset, mode="r+")
    assert myelin_ds.data.dtype == np.uint8

    assert(downsample_xy is not None)

    print("WORKER: Running with context %s" % os.environ['DAISY_CONTEXT'])
    client_scheduler = daisy.Client()

    os.environ["LAZYFLOW_THREADS"] = str(lazyflow_num_threads)
    os.environ["LAZYFLOW_TOTAL_RAM_MB"] = str(lazyflow_mem)

    args = ilastik_main.parser.parse_args([])
    args.headless = True
    args.project = ilastik_project_path

    shell = ilastik_main.main(args)
    assert isinstance(shell.workflow,
        (PixelClassificationWorkflow, AutocontextTwoStage))

    # Obtain the training operator
    opPixelClassification = shell.workflow.pcApplet.topLevelOperator

    # Sanity checks
    assert len(opPixelClassification.InputImages) > 0
    assert opPixelClassification.Classifier.ready()

    while True:
        block = client_scheduler.acquire_block()
        if block is None:
            break

        predict_myelin_2d(
            block,
            raw_ds,
            myelin_ds,
            downsample_xy)

        client_scheduler.release_block(block, ret=0)

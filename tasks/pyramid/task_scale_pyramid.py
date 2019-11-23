import argparse
import daisy
import numpy as np
import re
import skimage.measure
import zarr
import pymongo
import sys
import json
import hashlib
import multiprocessing
import logging

# monkey-patch os.mkdirs, due to bug in zarr
import os
prev_makedirs = os.makedirs


def makedirs(name, mode=0o777, exist_ok=False):
    # always ok if exists
    return prev_makedirs(name, mode, exist_ok=True)


os.makedirs = makedirs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def downscale_block(in_array, out_array, factor, block, completion_db):

    factor = tuple(factor)
    dims = len(factor)
    in_data = in_array.to_ndarray(block.read_roi, fill_value=0)

    in_shape = daisy.Coordinate(in_data.shape[-dims:])
    assert in_shape.is_multiple_of(factor)

    n_channels = len(in_data.shape) - dims
    if n_channels >= 1:
        factor = (1,)*n_channels + factor

    if in_data.dtype == np.uint64:
        slices = tuple(slice(k//2, None, k) for k in factor)
        out_data = in_data[slices]
    else:
        out_data = skimage.measure.block_reduce(in_data, factor, np.mean)

    try:
        out_array[block.write_roi] = out_data
    except Exception:
        print("Failed to write to %s" % block.write_roi)
        raise

    if completion_db is not None:
        document = {
            'block_id': block.block_id
        }
        completion_db.insert(document)

    return 0


def __run_worker():

    print(sys.argv)
    config_file = sys.argv[2]
    with open(config_file, 'r') as f:
        run_config = json.load(f)
    print(run_config)
    for key in run_config:
        globals()['%s' % key] = run_config[key]
    logger.debug("WORKER: Running with context %s"%os.environ['DAISY_CONTEXT'])
    client_scheduler = daisy.Client()
    db_client = pymongo.MongoClient(db_host)
    db = db_client[db_name]
    completion_db = db[completion_db_name] 

    in_array = daisy.open_ds(in_file, in_ds)
    out_array = daisy.open_ds(out_file, out_ds, mode='r+')

    while True:
        block = client_scheduler.acquire_block()
        if block is None:
            break

        logger.debug("Processing", block)

        downscale_block(
            in_array,
            out_array,
            factor,
            block,
            completion_db)

        client_scheduler.release_block(block, ret=0)


def check_block(
        block,
        vol_ds,
        completion_db
        ):

    write_roi = vol_ds.roi.intersect(block.write_roi)
    if write_roi.empty():
        return True

    if completion_db.count({'block_id': block.block_id}) >= 1:
        return True

    # quarter = (write_roi.get_end() - write_roi.get_begin()) / 4

    # # check values of center and nearby voxels
    # # if np.sum(vol_ds[write_roi.get_begin() + quarter*1]): return True
    # if np.sum(vol_ds[write_roi.get_begin() + quarter*2]): return True
    # # if np.sum(vol_ds[write_roi.get_begin() + quarter*3]): return True

    return False

config_file = None
# context_str = None
# new_actor_cmd = None

launch_process_cmd = multiprocessing.Manager().dict()


def new_actor():

    global config_file
    global launch_process_cmd

    context_str = os.environ['DAISY_CONTEXT']
    script_file = os.path.realpath(__file__)
    new_actor_cmd = 'python %s run_worker %s' % (script_file, config_file)

    if 'context_str' not in launch_process_cmd:
        launch_process_cmd['context_str'] = context_str
        launch_process_cmd['new_actor_cmd'] = new_actor_cmd

    sys.argv = [
        script_file,
        'run_worker',
        config_file,
        ]

    __run_worker()

    # periodic_callback(None)


def periodic_callback(args):

    global launch_process_cmd
    if "context_str" in launch_process_cmd:
        print("Submit command: DAISY_CONTEXT={} {}".format(
                launch_process_cmd["context_str"], launch_process_cmd["new_actor_cmd"]))
                # ' '.join(new_actor_cmd)))


def downscale(
        in_array, out_array, factor, write_size, roi, completion_db=None,
        in_file=None,
        in_ds=None,
        out_file=None,
        out_ds=None,
        num_workers=48,
        db_host=None,
        db_name=None,
        completion_db_name=None,
        ):

    global config_file

    print("Downsampling by factor %s" % (factor,))

    dims = in_array.roi.dims()
    block_roi = daisy.Roi((0,)*dims, write_size)

    config = {
        'in_file': in_file,
        'in_ds': in_ds,
        'out_file': out_file,
        'out_ds': out_ds,
        'factor': factor,
        'db_host': db_host,
        'db_name': db_name,
        'completion_db_name': completion_db_name,
    }
    config_str = ''.join(['%s' % (v,) for v in config.values()])
    config_hash = abs(int(hashlib.md5(config_str.encode()).hexdigest(), 16))
    try:
        os.makedirs('.run_configs')
    except Exception:
        pass
    config_file = os.path.join(
        '.run_configs', '%s_%d.config' % ("task_scale_pyramid", config_hash))
    with open(config_file, 'w') as f:
        json.dump(config, f)

    print("Processing ROI %s with blocks %s" % (roi, block_roi))

    precheck = lambda b: True
    postcheck = lambda b: True

    if completion_db is not None:
        precheck = lambda b: check_block(b, out_array, completion_db)

    daisy.run_blockwise(
        roi,
        block_roi,
        block_roi,
        process_function=new_actor,
        # process_function=lambda b: downscale_block(
        #     in_array,
        #     out_array,
        #     factor,
        #     b,
        #     completion_db),
        check_function=(precheck, postcheck),
        read_write_conflict=False,
        num_workers=num_workers,
        max_retries=0,
        fit='shrink',
        periodic_callback=periodic_callback)


def create_scale_pyramid(
        in_file, in_ds_name, scales, chunk_shape,
        num_workers=48,
        roi=None, sub_roi=None,
        db_host=None, db_name=None):

    ds = zarr.open(in_file)

    # make sure in_ds_name points to a dataset
    try:
        daisy.open_ds(in_file, in_ds_name)
    except Exception:
        raise RuntimeError("%s does not seem to be a dataset" % in_ds_name)

    db = None
    if db_host is not None:

        if db_name is None:
            db_name = os.path.split(in_file)[1].split('.')[0]

        db_client = pymongo.MongoClient(db_host)

        print(db_name)

        if db_name in db_client.database_names():
            i = input("Reset completion stats for %s before running? Yes/[No]" % db_name)
            assert i == "Yes" or i == "No"
            if i == "Yes":
                db_client.drop_database(db_name)

        db = db_client[db_name]

    initial_scale = 0
    m = re.match("(.+)/s(\d+)", in_ds_name)
    if m:

        initial_scale = int(m.group(2))
        ds_name = in_ds_name
        in_ds_name = m.group(1)

    elif not in_ds_name.endswith('/s0'):

        ds_name = in_ds_name + '/s0'

        i = input("Moving %s to %s? Yes/[No]" % (in_ds_name, ds_name))
        if i != "Yes":
            exit(0)
        ds.store.rename(in_ds_name, in_ds_name + '__tmp')
        ds.store.rename(in_ds_name + '__tmp', ds_name)

    else:

        ds_name = in_ds_name
        in_ds_name = in_ds_name[:-3]

    print("Scaling %s by a factor of %s" % (in_file, scales))

    prev_array = daisy.open_ds(in_file, ds_name)

    if chunk_shape is not None:
        chunk_shape = daisy.Coordinate(chunk_shape)
    else:
        chunk_shape = daisy.Coordinate(prev_array.data.chunks)
        print("Reusing chunk shape of %s for new datasets" % (chunk_shape,))

    if prev_array.n_channel_dims == 0:
        num_channels = 1
    elif prev_array.n_channel_dims == 1:
        num_channels = prev_array.shape[0]
    else:
        raise RuntimeError(
            "more than one channel not yet implemented, sorry...")

    if roi is None:
        ds_roi = prev_array.roi
        schedule_roi = prev_array.roi
    else:
        assert False, "Untested"
        ds_roi = roi
        schedule_roi = roi

    if sub_roi is not None:
        assert roi is None
        schedule_roi = sub_roi

    prev_ds_name = ds_name

    for scale_num, scale in enumerate(scales):

        if scale_num + 1 > initial_scale:

            try:
                scale = daisy.Coordinate(scale)
            except Exception:
                scale = daisy.Coordinate((scale,)*chunk_shape.dims())

            next_voxel_size = prev_array.voxel_size*scale
            next_write_size = chunk_shape*next_voxel_size

            next_schedule_roi = schedule_roi.snap_to_grid(
                next_voxel_size,
                mode='grow')

            next_ds_roi = ds_roi.snap_to_grid(
                next_voxel_size,
                mode='grow')
            if sub_roi is not None:
                # with sub_roi, the coordinates are absolute
                # so we'd need to align total_roi to the write size too
                next_ds_roi = next_ds_roi.snap_to_grid(
                    next_write_size, mode='grow')
                next_schedule_roi = next_schedule_roi.snap_to_grid(
                    next_write_size, mode='grow')

            print("Next voxel size: %s" % (next_voxel_size,))
            print("next_ds_roi: %s" % next_ds_roi)
            print("next_schedule_roi: %s" % next_schedule_roi)
            print("Next chunk size: %s" % (next_write_size,))

            next_ds_name = in_ds_name + '/s' + str(scale_num + 1)
            print("Preparing %s" % (next_ds_name,))

            try:
                next_array = daisy.open_ds(in_file, next_ds_name, mode='r+')
                assert next_array.roi.contains(next_schedule_roi)
            except:
                next_array = daisy.prepare_ds(
                    in_file,
                    next_ds_name,
                    total_roi=next_ds_roi,
                    voxel_size=next_voxel_size,
                    write_size=next_write_size,
                    dtype=prev_array.dtype,
                    num_channels=num_channels)

            completion_db = None
            if db:
                collection_name = 'pyramid_' + str(scale_num + 1)

                if collection_name not in db.list_collection_names():
                    completion_db = db[collection_name]
                    completion_db.create_index(
                        [('block_id', pymongo.ASCENDING)],
                        name='block_id')
                else:
                    completion_db = db[collection_name]

            downscale(
                prev_array,
                next_array,
                scale,
                next_write_size,
                next_schedule_roi,
                num_workers=num_workers,
                completion_db=completion_db,
                in_file=in_file,
                in_ds=prev_ds_name,
                out_file=in_file,
                out_ds=next_ds_name,
                db_host=db_host,
                db_name=db_name,
                completion_db_name=collection_name,
                )

            prev_array = next_array
            prev_ds_name = next_ds_name


if __name__ == "__main__":

    if sys.argv[1] == 'run_worker':
        __run_worker()
        exit(0)

    parser = argparse.ArgumentParser(
        description="Create a scale pyramide for a zarr/N5 container.")

    parser.add_argument(
        '--file',
        '-f',
        type=str,
        help="The input container")
    parser.add_argument(
        '--ds',
        '-d',
        type=str,
        help="The name of the dataset")
    parser.add_argument(
        '--scales',
        '-s',
        nargs='*',
        type=int,
        required=True,
        help="The downscaling factor between scales")
    parser.add_argument(
        '--chunk_shape',
        '-c',
        nargs='*',
        type=int,
        default=None,
        help="The size of a chunk in voxels")
    parser.add_argument(
        "--max_voxel_count", type=int, help='zyx size in pixel',
        default=256*1024)

    args = parser.parse_args()

    create_scale_pyramid(args.file, args.ds, args.scales, args.chunk_shape,
        args.max_voxel_count)

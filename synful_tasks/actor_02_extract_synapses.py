import copy
import json
import os
import logging
import numpy as np
import sys
import daisy
import pymongo
import time
import functools
# import math
import synapse
import detection
from database_synapses import SynapseDatabase
from database_superfragments import SuperFragmentDatabase, SuperFragment
from daisy import Coordinate

from downscale import convert_affs

sys.path.insert(0, '/n/groups/htem/Segmentation/tmn7/segway.synapse.area')
from segway.synapse.area.realign import Realigner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# logger.setLevel('WARNING')
logger.setLevel('INFO')
# logger.setLevel('DEBUG')

# debug = True
debug = False

# def to_abs_ng_coord(zyx, block_offset):
#     zyx += block_offset
#     return (zyx[2]/40, zyx[1]/4, zyx[0]/4)
def to_abs_xyz_ng_coord(zyx, block_offset):
    zyx = zyx.copy()
    zyx += block_offset
    # return (zyx[0]/4, zyx[1]/4, zyx[2]/40)
    return (int(zyx[2]/4),
            int(zyx[1]/4),
            int(zyx[0]/40))

def __create_unique_syn_id(zyx):

    id = 0
    binary_str = []
    for i in zyx:
        binary_str.append("{0:021b}".format(int(i)))
    id = int(''.join(binary_str), 2)

    return id


def __create_syn_ids(zyx_list):

    ret = []
    for zyx in zyx_list:
        ret.append(__create_unique_syn_id(zyx))

    assert(len(set(ret)) == len(ret))  # make sure that we created unique IDs
    return ret


def __create_syn_locations(predicted_syns, target_sites):

    if len(predicted_syns) != len(target_sites):
        print("ERROR: pre and post synaptic site do not have same length!")
        print("Synapses location was not created!")

    else:
        loc_zyx =[]
        for i in range(len(predicted_syns)):
            loc_zyx.append((predicted_syns[i]+target_sites[i])/2)

        return loc_zyx
           
def extract_synapses(ind_pred_ds,
                     dir_pred_ds,
                     segment_ds,
                     parameters,
                     block,
                     prediction_mode,
                     remove_z_dir,
                     d_vector_scale,
                     affs_ds=None,
                     local_realigner=None,
                     local_alignment_offsets_xy=None,
                     ):
    """Extract synapses from the block and write in the DB"""
    ##### EXTRACT SYNAPSES
    start_time = time.time()

    read_roi = ind_pred_ds.roi.intersect(block.read_roi)
    logger.debug('read_roi: %s' % read_roi)


    affs_ndarray = None
    if affs_ds:
        read_roi = read_roi.intersect(affs_ds.roi)
        logger.debug(f'reducing roi to {read_roi} because of affs roi {affs_ds.roi}')
        # affs_array = affs_ds.to_ndarray(roi=read_roi)
        affs_array = affs_ds[read_roi]
        affs_array = convert_affs(affs_array, ind_pred_ds.voxel_size)
        affs_ndarray = affs_array.to_ndarray()

    zchannel = ind_pred_ds.to_ndarray(roi=read_roi)

    assert read_roi.contains(block.write_roi)

    if local_realigner is not None:
        local_realigner.set_local_offset(read_roi.get_offset())

    if len(zchannel.shape) == 4:
        zchannel = np.squeeze(zchannel[3, :])
    if zchannel.dtype == np.uint8:
        zchannel = zchannel.astype(np.float32)
        zchannel /= 255.  # Convert to float
        logger.debug('Rescaling z channel with 255')

    voxel_size = np.array(ind_pred_ds.voxel_size)
    predicted_locs, predicted_props = detection.find_locations(
                                                    zchannel, parameters,
                                                    voxel_size,
                                                    score_threshold=parameters.score_thr,
                                                    affs_ndarray=affs_ndarray,
                                                    local_realigner=local_realigner,
                                                    local_alignment_offsets_xy=local_alignment_offsets_xy,
                                                    )

    # for l, p in zip(predicted_locs, predicted_props):
    #     print(to_abs_xyz_ng_coord(l, read_roi.get_begin()))
    #     for k in p:
    #         print(f'{k}: {p[k]}')
    #     print('')

    # Load direction vectors and find target location
    dirmap = dir_pred_ds.to_ndarray(roi=read_roi)

    # Before rescaling, convert back to float
    dirmap = dirmap.astype(np.float32)
    if 'scale' in dir_pred_ds.data.attrs:
        scale = dir_pred_ds.data.attrs['scale']
        dirmap = dirmap * 1. / scale
    else:
        logger.warning(
            'Scale attribute of dir vectors not set. Assuming dir vectors unit: nm, max value {}'.format(
                np.max(dirmap)))

    find_targets = functools.partial(detection.find_targets,
                                     dirvectors=dirmap,
                                     voxel_size=voxel_size,
                                     remove_z_dir=remove_z_dir,
                                     d_vector_scale=d_vector_scale,
                                     )

    predicted_partner_locs = find_targets(predicted_locs)

    # correct pre/post locs for different prediction modes
    if debug:
        assert prediction_mode == "cleft_to_pre"
    if prediction_mode == "cleft_to_pre":
        # post is actually cleft
        predicted_cleft_locs = predicted_locs
        predicted_pre_locs = predicted_partner_locs
        predicted_post_locs = find_targets(predicted_cleft_locs, reverse_dir=True)
    elif prediction_mode == "pre_to_post":
        predicted_pre_locs = predicted_locs
        predicted_post_locs = predicted_partner_locs
        predicted_cleft_locs = __create_syn_locations(predicted_pre_locs, predicted_post_locs)
    elif prediction_mode == "post_to_pre":
        predicted_post_locs = predicted_locs
        predicted_pre_locs = predicted_partner_locs
        predicted_cleft_locs = __create_syn_locations(predicted_pre_locs, predicted_post_locs)
        pass
    else:
        raise RuntimeError(f'Invalid prediction_mode: {prediction_mode}')

    # Synapses need to be shifted to the global ROI
    # (currently aligned to block.roi)
    for loc in predicted_post_locs:
        loc += np.array(read_roi.get_begin())
    for loc in predicted_pre_locs:
        loc += np.array(read_roi.get_begin())
    for loc in predicted_cleft_locs:
        loc += np.array(read_roi.get_begin())

    # because the synapse_id is created based on cleft loc, only keep those within `write_roi`
    # also, we cannot keep synapses with pre/post locs outside of `read_roi`; make sure that
    # it is big enough in the config
    filt_ind = []
    for i, (post_loc, pre_loc, cleft_loc) in enumerate(zip(predicted_post_locs, predicted_pre_locs, predicted_cleft_locs)):
        if not block.write_roi.contains(cleft_loc):
            logger.debug(f'{cleft_loc} not in block.write_roi')
            continue
        if not read_roi.contains(pre_loc) or not read_roi.contains(post_loc):
            logger.debug(f'pre_loc {pre_loc} or post_loc {post_loc} not in read_roi')
            continue
        filt_ind.append(i)
    logger.debug(f'read_roi: {read_roi}')
    logger.debug(f'write_roi: {block.write_roi}')
    logger.debug(f'{len(predicted_cleft_locs)-len(filt_ind)}/{len(predicted_cleft_locs)} got filtered out')

    post_syn_locs = list(np.array(predicted_post_locs)[filt_ind])
    pre_syn_locs = list(np.array(predicted_pre_locs)[filt_ind])
    # cleft_syn_locs = list(np.array(predicted_cleft_locs)[filt_ind])
    props = list(np.array(predicted_props)[filt_ind])

    segment_ds = segment_ds[read_roi]
    segment_ds.materialize()

    # Superfragments IDs
    ids_sf_pre = []
    for pre_syn in pre_syn_locs:
        pre_syn = Coordinate(pre_syn)
        pre_super_fragment_id = segment_ds[pre_syn]
        assert pre_super_fragment_id is not None
        ids_sf_pre.append(pre_super_fragment_id)

    ids_sf_post = []
    for post_syn in post_syn_locs:
        post_syn = Coordinate(post_syn)
        post_super_fragment_id = segment_ds[post_syn]
        assert post_super_fragment_id is not None
        ids_sf_post.append(post_super_fragment_id)

    assert len(pre_syn_locs) == len(post_syn_locs)
    assert len(pre_syn_locs) == len(props)

    # filter false positives
    pre_syns_f = []
    post_syns_f = []
    props_f = []
    i_f = [] # indices to consider
    for i in range(len(ids_sf_pre)):
        if ids_sf_pre[i] == ids_sf_post[i]:
            continue
        if ids_sf_pre[i] == 0 or ids_sf_post[i] == 0:
            continue
        pre_syns_f.append(pre_syn_locs[i])
        post_syns_f.append(post_syn_locs[i])
        props_f.append(props[i])
        i_f.append(i)

    ids_sf_pre = list(np.array(ids_sf_pre)[i_f])
    ids_sf_post = list(np.array(ids_sf_post)[i_f])
    # Create xyz locations
    cleft_syn_locs = __create_syn_locations(pre_syns_f, post_syns_f)
    # Create IDs for synpses from volume coordinates
    ids = __create_syn_ids(cleft_syn_locs)  # make ID based on cleft for uniqueness

    synapses = synapse.create_synapses(pre_syns_f, post_syns_f,
                                   props=props_f, ID=ids, zyx=cleft_syn_locs,
                                   ids_sf_pre=ids_sf_pre,
                                   ids_sf_post=ids_sf_post)

    if debug:
        for syn in synapses:
            print(syn)

    return synapses 


def extract_superfragments(synapses, write_roi):

    superfragments = {}
    for syn in synapses:

        pre_partner_id = int(syn.id_superfrag_pre)
        post_partner_id = int(syn.id_superfrag_post)

        if write_roi.contains(Coordinate(syn.location_pre)):

            if pre_partner_id not in superfragments:
                superfragments[pre_partner_id] = SuperFragment(id=pre_partner_id)

            superfragments[pre_partner_id].syn_ids.append(syn.id)
            superfragments[pre_partner_id].post_partners.append(post_partner_id)

        if write_roi.contains(Coordinate(syn.location_post)):

            if post_partner_id not in superfragments:
                superfragments[post_partner_id] = SuperFragment(id=post_partner_id)

            superfragments[post_partner_id].syn_ids.append(syn.id)
            superfragments[post_partner_id].pre_partners.append(pre_partner_id)

    superfragments_list = [superfragments[item] for item in superfragments]
    for sf in superfragments_list:
        sf.finalize()

    return superfragments_list


if __name__ == "__main__":

    print(sys.argv)
    config_file = sys.argv[1]
    with open(config_file, 'r') as f:
        run_config = json.load(f)

    for key in run_config:
        globals()['%s' % key] = run_config[key]

    '''Illaria TODO
        1. Check for different thresholds
            make them daisy.Parameters
            see the plotting script for these parameters and thesholds: https://github.com/htem/segway/tree/master/synapse_evaluation
    '''

    db_client = pymongo.MongoClient(db_host)
    db = db_client[db_name]
    completion_db = db[completion_db_name]
    logger.debug(f"db_name: {db_name}")
    logger.debug(f"db_host: {db_host}")
    logger.debug(f"db collection names: {db_col_name_syn}, {db_col_name_sf}")
    logger.debug(f"super_fragments_file: {super_fragments_file}")
    logger.debug(f"super_fragments_dataset: {super_fragments_dataset}")
    logger.debug(f"syn_indicator_file: {syn_indicator_file}")
    logger.debug(f"syn_indicator_dataset: {syn_indicator_dataset}")
    logger.debug(f"syn_dir_file: {syn_dir_file}")
    logger.debug(f"syn_dir_dataset: {syn_dir_dataset}")
    logger.debug(f"score_threshold: {score_threshold}")
    logger.debug(f"prediction_mode: {prediction_mode}")
    logger.debug(f"remove_z_dir: {remove_z_dir}")
    logger.debug(f"d_vector_scale: {d_vector_scale}")

    if debug:
        assert prediction_mode == "cleft_to_pre"
        assert remove_z_dir is True

    parameters = detection.SynapseExtractionParameters(
        extract_type=extract_type,
        cc_threshold=cc_threshold,
        loc_type=loc_type,
        score_thr=score_threshold,
        score_type=score_type,
        nms_radius=None,
    )

    logger.info("WORKER: Running with context %s"%os.environ['DAISY_CONTEXT'])
    client_scheduler = daisy.Client()

    syn_db = SynapseDatabase(db_name, db_host, db_col_name_syn,
                 mode='r+')
    superfrag_db = SuperFragmentDatabase(db_name, db_host, db_col_name_sf,
                 mode='r+')
    ind_pred_ds = daisy.open_ds(syn_indicator_file, syn_indicator_dataset, 'r') 
    dir_pred_ds = daisy.open_ds(syn_dir_file, syn_dir_dataset, 'r')
    segment_ds = daisy.open_ds(super_fragments_file, super_fragments_dataset, 'r')
    affs_ds = None
    if affs_file or affs_dataset:
        assert affs_file and affs_dataset
        affs_ds = daisy.open_ds(affs_file, affs_dataset, 'r')

    local_realigner = None
    if raw_file or raw_dataset:
        assert raw_file and raw_dataset
        local_realigner = Realigner(raw_file, raw_dataset,
                                    # xy_context_nm=realignment_xy_context_nm,
                                    xy_context_nm=0,
                                    xy_stride_nm=realignment_xy_stride_nm,
                                    )
        raw_ds = daisy.open_ds(raw_file, raw_dataset, 'r')

    else:
        asdf

    while True:
        block = client_scheduler.acquire_block()
        if block is None:
            break

        logging.info("Running synapse extraction for block %s" % block)

        realign_roi = raw_ds.roi.intersect(block.read_roi)
        # local_alignment_offsets_xy = local_realigner.multipass_realign(block.write_roi, [1, 2, 3])
        local_alignment_offsets_xy = local_realigner.multipass_realign(realign_roi, [1, 2, 3, 4, 5])

        synapses = extract_synapses(ind_pred_ds,
                                    dir_pred_ds,
                                    segment_ds,
                                    parameters,
                                    block,
                                    prediction_mode=prediction_mode,
                                    remove_z_dir=remove_z_dir,
                                    d_vector_scale=d_vector_scale,
                                    affs_ds=affs_ds,
                                    # local_realigner=local_realigner,
                                    local_realigner=None,
                                    local_alignment_offsets_xy=local_alignment_offsets_xy,
                                    )

        superfragments = extract_superfragments(synapses, block.write_roi)

        if debug:
            # FOR debug PURPOSES, DON'T RETURN THE BLOCK AND JUST QUIT
            time.sleep(1)
            sys.exit(1)

        syn_db.write_synapses(synapses)
        superfrag_db.write_superfragments(superfragments)

        # write block completion
        document = {
            'block_id': block.block_id,
        }
        completion_db.insert(document)

        client_scheduler.release_block(block, ret=0)

    print("NUM SYNAPSES: ", syn_db.synapses.count())
    print("NUM SUPERFRAGMENTS: ", superfrag_db.superfragments.count())

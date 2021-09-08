import logging
import sys

import numpy as np
import numpy.ma as ma
from scipy import ndimage
from skimage import measure, morphology
import scipy

from nms import find_maxima

sys.path.insert(0, '/n/groups/htem/Segmentation/tmn7/segway.synapse.area')
from segway.synapse.area.extract import SynapseArea

logger = logging.getLogger(__name__)


class SynapseExtractionParameters(object):
    '''

    Args:

        extract_type (``string``, optional):

            How to detect synapse candidate locations. CC --> For each connected
            component of the thresholded input array, one location is extracted.

        cc_threshold (``float``, optional):

            Where to threshold input array to extract connected component.

        loc_type (``string``, optional):

            How to extract location from connected component:
            edt --> euclidean distance transform.

        score_thr (``string``, optional):

            Only consider location with a score value greater
            than this threshold.

        score_type (``string``, optional):

            How to calculate the score. Possible options: sum, mean, max, count.


    '''

    def __init__(
            self,
            extract_type='cc',
            cc_threshold=None,
            loc_type='edt',
            # How to extract location from blob: edt --> euclidean distance transform
            score_thr=None,  # If locs should be filtered with threshold
            score_type=None,  # What kind of score to use.
            nms_radius=None
    ):
        # assert extract_type == 'cc', 'Synapse Detection currently only ' \
        #                              'implemented with option cc'  # TODO: Implement nms
        if extract_type == 'nms':
            assert nms_radius is not None
        self.extract_type = extract_type
        self.cc_threshold = cc_threshold if extract_type == 'cc' else None
        self.loc_type = loc_type if extract_type == 'cc' else None
        self.score_type = score_type if extract_type == 'cc' else None
        self.score_thr = score_thr
        self.nms_radius = nms_radius if extract_type == 'nms' else None


def __from_labels_to_edt(labels, voxel_size):
    boundaries = __find_boundaries(labels)
    print(boundaries)
    boundaries = 1.0 - boundaries
    distances = ndimage.distance_transform_edt(
        boundaries,
        sampling=tuple(float(v) / 2 for v in voxel_size))
    distances = distances.astype(np.float32)

    # restore original shape
    downsample = (slice(None, None, 2),) * len(voxel_size)
    distances = distances[downsample]
    return distances


def __find_boundaries(labels):
    # labels: 1 1 1 1 0 0 2 2 2 2 3 3       n
    # shift :   1 1 1 1 0 0 2 2 2 2 3       n - 1
    # diff  :   0 0 0 1 0 1 0 0 0 1 0       n - 1
    # bound.: 00000001000100000001000      2n - 1

    dims = len(labels.shape)
    in_shape = labels.shape
    out_shape = tuple(2 * s - 1 for s in in_shape)

    boundaries = np.zeros(out_shape, dtype=np.bool)

    for d in range(dims):
        shift_p = [slice(None)] * dims
        shift_p[d] = slice(1, in_shape[d])

        shift_n = [slice(None)] * dims
        shift_n[d] = slice(0, in_shape[d] - 1)

        diff = (labels[tuple(shift_p)] - labels[tuple(shift_n)]) != 0

        target = [slice(None, None, 2)] * dims
        target[d] = slice(1, out_shape[d], 2)

        boundaries[tuple(target)] = diff

    return boundaries


def __from_probmap_to_labels(probmap, threshold):
    """Thresholds an intensity map and find connected components.

    Args:
        probmap (np.array): The original array with probabilities.
        threshold (int/float): threshold

    Returns:
        regions:
        res: numpy array in which each disconnected region has a unique ID.

    """
    res = np.zeros_like(probmap)
    res[probmap > threshold] = 1
    res, num_labels = ndimage.label(res)
    regions = measure.regionprops(res)
    return regions, res

from daisy import Coordinate
from daisy import Roi as DaisyRoi
from daisy import Array as DaisyArray
import math

# def __find_bb_boundary(slice_array, from_point, guess):

#     # don't calculate for very small shapes
#     for k in slice_array.shape:
#         if k <= 2:
#             return from_point

#     from_point = Coordinate(from_point)
#     guess = Coordinate(guess)
#     if guess == from_point:
#         return guess

#     min_guess = from_point
#     bbox = DaisyRoi((0, 0), tuple(k for k in slice_array.shape))
#     slice_array = DaisyArray(slice_array, bbox, voxel_size=(1, 1))

#     # debug = True
#     # debug = False

#     # if debug:
#     #     print('bbox:', bbox)
#     #     print('from_point:', from_point)
#     #     print('guess:', guess)

#     # find max_guess
#     max_guess = guess
#     assert max_guess != from_point
#     while bbox.contains(max_guess) and slice_array[max_guess]:
#         delta = max_guess - from_point
#         max_guess = from_point + delta*2

#     guess = (from_point + max_guess) / 2
#     last_guess = guess
#     # i = 0
#     while True:
#         if not bbox.contains(guess) or not slice_array[guess]:
#             # too high
#             max_guess = guess
#             # print('too high')
#         else:
#             assert slice_array[guess]
#             min_guess = guess
#             # print('too low')
#         guess = (min_guess+max_guess)/2
#         guess = Coordinate(tuple(int(c) for c in guess))
#         # if debug:
#         #     print('min_guess:', min_guess)
#         #     print('max_guess:', max_guess)
#         #     print('next guess:', guess)
#         #     print()
#         #     i += 1
#         #     if i > 10:
#         #         asdf
#         if max_guess == min_guess or guess == last_guess:
#             # DONE
#             break
#         last_guess = guess
#     return guess


# def __get_minor_axis_center(slice_array, centroid, orientation, minor_axis_length):

#     # don't calculate for very small shapes
#     for k in slice_array.shape:
#         if k <= 2:
#             return centroid
#     # determine lower boundary and higher boundary
#     y0, x0 = centroid
#     pos_guess_x = x0 - math.sin(orientation) * minor_axis_length
#     neg_guess_x = x0 + math.sin(orientation) * minor_axis_length
#     pos_guess_y = y0 - math.cos(orientation) * minor_axis_length
#     neg_guess_y = y0 + math.cos(orientation) * minor_axis_length

#     pos_guess = (pos_guess_y, pos_guess_x)
#     pos_bound = __find_bb_boundary(slice_array, centroid, pos_guess)
#     neg_guess = (neg_guess_y, neg_guess_x)
#     neg_bound = __find_bb_boundary(slice_array, centroid, neg_guess)

#     # correction for rounding down
#     pos_bound = pos_bound + (1, 1)
#     center = (pos_bound + neg_bound) / 2
#     return center


def __get_center_from_skeleton(slice_array):
    # skeleton = morphology.skeletonize(slice_array).astype(np.uint8)
    skeleton = morphology.skeletonize(slice_array)
    # print(skeleton)
    # print((skeleton).astype(np.uint8))
    # skeleton_centroid = scipy.ndimage.measurements.center_of_mass(skeleton)
    verts = np.argwhere(skeleton)
    centroid = np.mean(verts, axis=0)
    # print(verts)
    # print(centroid)
    min_vert = None
    min_dist = sys.maxsize
    for v in verts:
        d = np.linalg.norm((v[0]-centroid[0], v[1]-centroid[1]))
        if d < min_dist:
            min_dist = d
            min_vert = v
    return min_vert


def __extract_slice_syn_properties(slice_array, threshold=0):
    # labels, _ = ndimage.label(slice_array > threshold)
    # props = measure.regionprops(labels, slice_array)
    # major_axis_length = props[0].major_axis_length

    # centroid = tuple(int(k) for k in props[0].centroid)
    # minor_axis_center = __get_minor_axis_center(
        # slice_array, centroid, props[0].orientation, props[0].minor_axis_length)

    # print(labels)
    skel_center = __get_center_from_skeleton(slice_array > threshold)

    # print('centroid:', centroid)
    # print('minor_axis_center:', minor_axis_center)
    # print('skel_center:', skel_center)
    # # print('skeleton_centroid:', skeleton_centroid)
    # print('props[0].orientation:', props[0].orientation)
    # print('props[0].minor_axis_length:', props[0].minor_axis_length)
    # print('props[0].major_axis_length:', props[0].major_axis_length)

    return {
        # 'major_axis_length': major_axis_length,
        # 'minor_axis_center': minor_axis_center,
        # 'centroid': centroid,
        'skel_center': skel_center,
    }


def __from_labels_to_locs(labels, regions, voxel_size,
                          intensity_vol=None,
                          score_vol=None,
                          score_type=None,
                          score_threshold=None,
                          affs_ndarray=None,
                          local_realigner=None,
                          local_alignment_offsets_xy=None,
                          ):
    """Function that extracts locations from connected components.

    Args:
        labels (np.array): The array with connecected components (each marked
        with an unique ID).

        regions (regionsprops.regions): The regionsprops extracted from labels.

        voxel_size (np.array): voxel size

        intensity_vol (np.array): an array with the same shape as labels.
        If given, the maxima of this array represent the locations. If this
        is set to None, edt is calculated for the connected component itself
        and used for location extraction.

        score_vol (np.array): array to use to calculate the score from.

        score_type (str): how to combine the score values.

    Returns:
        locs: list of locations in world units.

    """

    locs = []
    props = []
    syn_areas = {}
    erosion_factors = [0, 1, 2]
    for i in erosion_factors:
        syn_area = SynapseArea(synapse_labels=labels,
                               regions=regions,
                               voxel_size=(40, 16, 16),
                               affs_ndarray=affs_ndarray,
                               local_realigner=local_realigner,
                               local_realignment=True,
                               local_alignment_offsets_xy=local_alignment_offsets_xy,
                               binary_erosion=i,
                               )
        syn_areas[i] = syn_area


    for reg in regions:
        prop = {}
        label_id = reg['label']
        z1, y1, x1, z2, y2, x2 = reg['bbox']
        z_length = abs(z1 - z2) + 1  # +1 to account for 1 section thickness

        crop = labels[z1:z2, y1:y2, x1:x2]
        reg_mask = crop != label_id
        bb_centroid = tuple(int(k) for k in reg.local_centroid)

        # Obtain score based on score_vol.
        if score_vol is not None:
            score_vol_crop = score_vol[z1:z2, y1:y2, x1:x2]
            score_crop = ma.masked_array(score_vol_crop, reg_mask)
            if score_type == 'sum':
                score = score_crop.sum()
            elif score_type == 'mean':
                score = score_crop.mean()
            elif score_type == 'max':
                score = score_crop.max()
            elif score_type == 'count':
                score = reg['area']
            else:
                raise RuntimeError('score not defined')

            if score_threshold and score < score_threshold:
                continue

            for erosion_factor in erosion_factors:
                syn_area = syn_areas[erosion_factor]
                syn_area.local_realignment = False
                prop[f'area_erode{erosion_factor}_no_realignment'] = syn_area.get_area(label_id, [
                    'mesh', 'ellipsoid_with_drift', 
                    ])
                syn_area.local_realignment = True
                prop[f'area_erode{erosion_factor}'] = syn_area.get_area(label_id, [
                    'pixel', 'skeleton', 'mesh', 'ellipsoid_with_drift'])

            prop['score'] = score
            prop['raw_pred_pix_count'] = reg['area']
            prop['z_length'] = z_length * voxel_size[0]
            crop_center_z = bb_centroid[0]
            # center_z = crop_center_z + z1
            # crop_center_z = center_z - z1
            score_crop = score_crop.filled(fill_value=0)
            center_slice = score_crop[crop_center_z]
            center_slice_props = __extract_slice_syn_properties(center_slice)
            # major_axis_length = center_slice_props['major_axis_length'] * voxel_size[1]

            loc_local = [
                crop_center_z,
                # center_slice_props['minor_axis_center'][0],
                # center_slice_props['minor_axis_center'][1]
                center_slice_props['skel_center'][0],
                center_slice_props['skel_center'][1]
                ]
            loc_abs = np.array(loc_local) + np.array([z1, y1, x1])
            # prop['major_axis_length'] = major_axis_length

            # if prop['raw_pred_pix_count'] == 2204:
            #     print(f'loc_abs: {loc_abs}')
            #     loc_abs = loc_abs * [40, 16, 16]
            #     loc_abs += [3000, 2048, 2048]
            #     loc_abs = loc_abs / [40, 4, 4]
            #     loc_abs = [loc_abs[2], loc_abs[1], loc_abs[0]]
            #     print(f'loc_abs: {loc_abs}')
            #     print(f'center_slice_props: {center_slice_props}')
            #     asdf
        locs.append(loc_abs * voxel_size)

        # convert datatypes for mongodb bson
        prop['z_length'] = int(prop['z_length'])
        prop['raw_pred_pix_count'] = int(prop['raw_pred_pix_count'])
        prop['score'] = float(prop['score'])

        props.append(prop)
    if score_vol is not None:
        assert len(locs) == len(props)
        return locs, props
    else:
        return locs


def find_locations(probmap, parameters,
                   voxel_size,
                   score_threshold=None,
                   affs_ndarray=None,
                   local_realigner=None,
                   local_alignment_offsets_xy=None,
                   ):
    """Function that extracts locations from an intensity / probability map.

    Args:
        probmap (np.array): Intensity array, higher value indicates presence
        of objects to be found.

        parameters (regionsprops.regions): synapse parameters

        voxel_size (np.array): voxel size

    Returns:

        locs: list of locations in world units.

    """
    voxel_size = np.array(voxel_size)
    props = {}
    regions, pred_labels = __from_probmap_to_labels(probmap,
                                                    parameters.cc_threshold)
    assert parameters.loc_type == 'edt', 'unknown loc_type option set: {}'.format(parameters.loc_type)
    pred_locs, props = __from_labels_to_locs(pred_labels,
                                                 regions,
                                                 voxel_size,
                                                 score_vol=probmap,
                                                 score_type=parameters.score_type,
                                                 score_threshold=score_threshold,
                                                 affs_ndarray=affs_ndarray,
                                                 local_realigner=local_realigner,
                                                 local_alignment_offsets_xy=local_alignment_offsets_xy,
                                                 )
    pred_locs = [loc.astype(np.int64) for loc in pred_locs]
    return pred_locs, props


def find_targets(source_locs, dirvectors,
                 voxel_size=[1., 1., 1.],
                 min_dist=0,
                 remove_z_dir=False,
                 d_vector_scale=None,
                 reverse_dir=False):
    """Function that finds target position based on a direction vector map.

    Args:
        source_locs (list): list with source locations in world units.
        dirvectors (np.array): map with [dim, source_locs.shape]
        voxel_size (np.array): voxel size
        min_dist (float/int): threshold to filter target locations based on
        the distance of dir vector. This modifies the source_locs input.
    Returns:
        locs: List of locations in world units.

    """
    target_locs = []
    distances = []
    for loc in source_locs:
        loc_voxel = (loc / voxel_size).astype(np.uint32)
        # print(loc)
        # print(loc_voxel)
        dirvector = dirvectors[:, loc_voxel[0], loc_voxel[1], loc_voxel[2]]
        if d_vector_scale:
            dirvector = [k*d_vector_scale for k in dirvector]
        if remove_z_dir:
            dirvector[0] = 0
        if not reverse_dir:
            target_loc = loc + dirvector
        else:
            target_loc = loc - dirvector
        target_loc = np.round(target_loc / voxel_size) # snap to voxel grid,
        # assuming MSE trained direction vector models.
        target_loc *= voxel_size
        target_locs.append(target_loc)
        dist = np.linalg.norm(np.array(list(loc)) - np.array(list(target_loc)))
        distances.append(dist)
    to_remove = []
    for ii, dist in enumerate(distances):
        if dist < min_dist:
            to_remove.append(ii)

    # Delete items in reversed order such that the indeces stay correct
    for index in sorted(to_remove, reverse=True):
        del source_locs[index]
        del target_locs[index]
    if len(distances) > 0:
        logger.debug('Average distance of synapses %0.2f' % np.mean(distances))
    logger.debug('Removed {} synapses because distance '
                 'smaller than {}'.format(len(to_remove), min_dist))
    target_locs = [loc.astype(np.int64) for loc in target_locs]
    return target_locs

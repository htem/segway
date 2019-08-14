import daisy
import neuroglancer
import sys
import numpy as np
import os
import json

from funlib.show.neuroglancer import add_layer
from segway import task_helper

neuroglancer.set_server_bind_address('0.0.0.0')

# user_configs, global_config = task_helper.parseConfigs(sys.argv[1:])
# f = global_config["Input"]["output_file"]
# raw_file = global_config["Input"]["raw_file"]

try:

    user_configs, global_config = task_helper.parseConfigs(sys.argv[1:])
    f = global_config["Input"]["output_file"]
    raw_file = global_config["Input"]["raw_file"]

except Exception as e:

    print(e)

    f = sys.argv[1]
    try:
        raw_file = os.environ['raw_file']
    except:
        try:
            # try to guess based on folder directory name
            # assume it's something like xxx/model/iteration/output.zarr
            vol_path = os.path.dirname(os.path.realpath(f))
            print(vol_path)
            index = -4 if vol_path[-1] == '/' else -3
            # index = -4
            vol = vol_path.split('/')[index]
            print(vol)
            if 'pl2' in vol:
                raw_file = '/n/groups/htem/temcagt/datasets/cb2/segmentation/segmented_volumes/cb2_pl2_181022.hdf'
            elif 'ml0' in vol:
                raw_file = '/n/groups/htem/temcagt/datasets/cb2/segmentation/segmented_volumes/cb2_ml_1204_cleaned.hdf'
            elif 'ml3' in vol:
                raw_file = '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/ml3_gt.zarr'
            elif 'ml4' in vol:
                raw_file = '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/gt/ml4_gt/ml4_gt.zarr'
            elif 'pl' in vol or 'ml' in vol:
                # pl = 3
                vol = vol.split('_')[0]
                raw_file = '/n/groups/htem/temcagt/datasets/cb2/segmentation/tri/cb2_gt/%s/cb2_gt_%s.zarr' % (vol, vol)
            elif 'cutout4' in vol:
                raw_file = '/n/groups/htem/temcagt/datasets/cb2/segmentation/tri/cb2_gt/synapse_gt/cutout4/cb2_synapse_cutout4.zarr'
            elif 'cutout9' in vol:
                raw_file = '/n/groups/htem/temcagt/datasets/cb2/segmentation/tri/cb2_gt/synapse_gt/cutout9/cb2_synapse_cutout9.zarr'
            elif 'cb2_synapse_cutout' in vol:
                cutout_n = vol[len("cb2_synapse_cutout"):]
                cutout_n = cutout_n.split('_')[0]
                raw_file = '/n/groups/htem/temcagt/datasets/cb2/segmentation/tri/cb2_gt/synapse_gt/cutout%s/cb2_synapse_cutout%s.zarr' % (cutout_n, cutout_n)
            elif 'synapse_cutout' in vol:
                cutout_n = vol[len("synapse_cutout"):]
                cutout_n = cutout_n.split('_')[0]
                raw_file = '/n/groups/htem/temcagt/datasets/cb2/segmentation/tri/cb2_gt/synapse_gt/cutout%s/cb2_synapse_cutout%s.zarr' % (cutout_n, cutout_n)
            elif 'myelin_cutout' in vol:
                cutout_n = vol[len("myelin_cutout"):]
                cutout_n = cutout_n.split('_')[0]
                raw_file = '/n/groups/htem/temcagt/datasets/cb2/segmentation/tri/cb2_gt/myelin/cutout%s/cb2_myelin_cutout%s.zarr' % (cutout_n, cutout_n)
            else:
                assert(0)
        except:
            assert(0)

if f[-1] == '/':
    f = f[:-1]
if raw_file[-1] == '/':
    raw_file = raw_file[:-1]

print("Opening %s..." % raw_file)
try:
    raw = daisy.open_ds(raw_file, 'volumes/raw')
except:
    raw = daisy.open_ds(raw_file, 'raw')


def add(s, a, name, shader=None):

    if shader == 'rgb':
        shader="""void main() { emitRGB(vec3(toNormalized(getDataValue(0)), toNormalized(getDataValue(1)), toNormalized(getDataValue(2)))); }"""

    kwargs = {}

    if shader == '255':
        shader="""void main() { emitGrayscale(float(getDataValue().value)); }"""

    if shader is not None:
        kwargs['shader'] = shader

    s.layers.append(
            name=name,
            layer=neuroglancer.LocalVolume(
                data=a.data,
                offset=a.roi.get_offset()[::-1],
                voxel_size=a.voxel_size[::-1]
            ),
            **kwargs)
    print(s.layers)


viewer = neuroglancer.Viewer()

with viewer.txn() as s:

    add_layer(s, raw, 'raw')

    add_layer(s, daisy.open_ds(f, 'volumes/affs'), 'affs', shader='rgb')
    add_layer(s, daisy.open_ds(f, 'volumes/myelin'), 'myelin', visible=False)
    add_layer(s, daisy.open_ds(f, 'volumes/fragments'), 'frag', visible=False)
    try: add_layer(s, daisy.open_ds(f, 'volumes/segmentation_0.100'), 'seg_100', visible=False)
    except: pass
    try: add_layer(s, daisy.open_ds(f, 'volumes/segmentation_0.200'), 'seg_200', visible=False)
    except: pass
    try: add_layer(s, daisy.open_ds(f, 'volumes/segmentation_0.300'), 'seg_300', visible=False)
    except: pass
    try: add_layer(s, daisy.open_ds(f, 'volumes/segmentation_0.400'), 'seg_400', visible=False)
    except: pass
    try: add_layer(s, daisy.open_ds(f, 'volumes/segmentation_0.500'), 'seg_500', visible=False)
    except: pass
    add_layer(s, daisy.open_ds(f, 'volumes/segmentation_0.600'), 'seg_600', visible=False)
    add_layer(s, daisy.open_ds(f, 'volumes/segmentation_0.700'), 'seg_700')
    add_layer(s, daisy.open_ds(f, 'volumes/segmentation_0.800'), 'seg_800', visible=False)
    add_layer(s, daisy.open_ds(f, 'volumes/segmentation_0.900'), 'seg_900', visible=False)

    segment = daisy.open_ds(f, 'volumes/segmentation_0.700')
    s.navigation.position.voxelCoordinates = np.flip(
        ((segment.roi.get_begin() + segment.roi.get_end()) / 2 / segment.voxel_size))


link = str(viewer)
print(link)
ip_mapping = [
    ['gandalf', 'catmaid3.hms.harvard.edu'],
    ['lee-htem-gpu0', '10.117.28.249'],
    ['lee-lab-gpu1', '10.117.28.82'],
    ]
for alias, ip in ip_mapping:
    if alias in link:
        print(link.replace(alias, ip))

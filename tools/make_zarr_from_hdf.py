import daisy
# import neuroglancer
import numpy as np
from PIL import Image
import sys
import json

config_f = sys.argv[1]
with open(config_f) as f:
    config = json.load(f)

hdf_ds = daisy.open_ds(
    config['hdf']['file'],
    config['hdf']['gt'],
    )

zarr_ds = daisy.prepare_ds(
    config['zarr']['file'],
    config['zarr']['gt'],
    hdf_ds.roi,
    hdf_ds.voxel_size,
    np.uint64,
    # write_size=raw_dir_shape,
    # num_channels=self.out_dims,
    # temporary fix until
    # https://github.com/zarr-developers/numcodecs/pull/87 gets approved
    # (we want gzip to be the default)
    compressor={'id': 'zlib', 'level': 5}
    )

# exit(0)

zarr_ds.data[:] = hdf_ds[hdf_ds.roi].to_ndarray()

exit(0)

#raw key
# gt = daisy.open_ds(f, 'volumes/labels/neuron_ids')
# lm = daisy.open_ds(f, 'volumes/labels/labels_mask')
# ul = daisy.open_ds(f, 'volumes/labels/unlabelled')

#path to labels
# f='output.zarr'


def add(s, a, name, shader=None):

    if shader == 'rgb':
        shader="""void main() { emitRGB(vec3(toNormalized(getDataValue(0)), toNormalized(getDataValue(1)), toNormalized(getDataValue(2)))); }"""

    kwargs = {}

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

    f, prepend = (f, "test")
    #add(s, daisy.open_ds(f, 'volumes/sparse_segmentation_0.5'), '%s_seg'%prepend)
    # add(s, daisy.open_ds(f, 'volumes/affs'), '%s_aff'%prepend)
    #add(s, daisy.open_ds(f, 'volumes/fragments'), '%s_frag'%prepend)
    #add(s, daisy.open_ds(f, 'volumes/segmentation_0.100'), '%s_seg_100'%prepend)
    #add(s, daisy.open_ds(f, 'volumes/segmentation_0.200'), '%s_seg_200'%prepend)
    #add(s, daisy.open_ds(f, 'volumes/segmentation_0.300'), '%s_seg_300'%prepend)
    #add(s, daisy.open_ds(f, 'volumes/segmentation_0.400'), '%s_seg_400'%prepend)
    # add(s, daisy.open_ds(f, 'volumes/sparse_segmentation_0.5'), '%s_seg_500'%prepend)

    add(s, raw, 'raw')
    # add(s, gt, 'gt')


print(viewer)
link = str(viewer)
# print("http://catmaid3.hms.harvard.edu:%d/v/%s/" % (forwarded_port, link.split('/')[-2]))

import daisy
import neuroglancer
import sys

import gt_tools

neuroglancer.set_server_bind_address('0.0.0.0')

arg = sys.argv[1]

if arg.endswith('.json'):
    config = gt_tools.load_config(arg, no_db=True)
    f = config["raw_file"]
elif arg.rstrip('/').endswith('.zarr'):
    f = arg
else:
    raise RuntimeError("Arg is neither .json nor .zarr")

if f[-1] == '/':
    f = f[:-1]

viewer = neuroglancer.Viewer()

with viewer.txn() as s:

    gt_tools.add_ng_layer(s, daisy.open_ds(f, 'volumes/raw'), 'raw')
    gt_tools.add_ng_layer(s, daisy.open_ds(f, 'volumes/labels/neuron_ids'), 'labels')
    gt_tools.add_ng_layer(s, daisy.open_ds(f, 'volumes/labels/myelin_gt'), 'myelin', shader='1')
    gt_tools.add_ng_layer(s, daisy.open_ds(f, 'volumes/labels/labels_mask'), 'labels_mask', shader='1')
    gt_tools.add_ng_layer(s, daisy.open_ds(f, 'volumes/labels/labels_mask2'), 'labels_mask2', shader='1')
    gt_tools.add_ng_layer(s, daisy.open_ds(f, 'volumes/labels/unlabeled'), 'unlabeled', shader='1')
    try:
        gt_tools.add_ng_layer(s, daisy.open_ds(f, 'volumes/labels/myelin_reject_mask'), 'myelin_reject_mask', shader='1')
    except:
        pass

gt_tools.print_ng_link(viewer)

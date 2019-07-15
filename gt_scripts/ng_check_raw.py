import daisy
import neuroglancer
import sys

import gt_tools

neuroglancer.set_server_bind_address('0.0.0.0')

if '.zarr' in sys.argv[1]:
    raw_file = sys.argv[1]
else:
    config = gt_tools.load_config(sys.argv[1])
    raw_file = config["raw_file"]

if raw_file[-1] == '/':
    raw_file = raw_file[:-1]

raw = daisy.open_ds(raw_file, 'volumes/raw')

viewer = neuroglancer.Viewer()

with viewer.txn() as s:

    gt_tools.add_ng_layer(s, raw, 'raw')

gt_tools.print_ng_link(viewer)

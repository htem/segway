import daisy
import neuroglancer
import sys
import os
import gt_tools

if '.zarr' in sys.argv[1]:
    raw_file = sys.argv[1]
else:
    config = gt_tools.load_config(sys.argv[1], no_db=True, no_zarr=True)
    raw_file = config["raw_file"]

if raw_file[-1] == '/':
    raw_file = raw_file[:-1]

raw = daisy.open_ds(raw_file, 'volumes/raw')

viewer = gt_tools.make_ng_viewer()

with viewer.txn() as s:

    gt_tools.add_ng_layer(s, raw, 'raw')

print("Raw ZARR at %s" % os.path.realpath(raw_file))

gt_tools.print_ng_link(viewer)

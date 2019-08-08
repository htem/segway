import daisy
import sys
import logging
import gt_tools

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

'''
07/29/19
'''

if __name__ == "__main__":

    config = gt_tools.load_config(sys.argv[1])

    if "CatmaidIn" in config:

        in_config = config["CatmaidIn"]
        assert in_config["roi_offset_encoding"] == "tile"

        z, y, x = daisy.Coordinate(in_config["roi_offset"]) * daisy.Coordinate(in_config["tile_shape"])
        print("x: %d y: %d z: %d" % (x, y, z))

    elif "ZarrIn" in config:

        in_config = config["ZarrIn"]
        assert in_config["roi_offset_encoding"] == "voxel"
        roi_offset = in_config["roi_offset"]

        print("x: %d y: %d z: %d" % (roi_offset[2], roi_offset[1], roi_offset[0]))



import json
import pymongo
import sys
import os

import neuroglancer

sys.path.insert(0, "/n/groups/htem/temcagt/datasets/cb2/segmentation/tri/cb2_segmentation/segway/tasks")
import task_helper


def get_db_names(config, file):

    db_name = config.get("db_name", None)
    db_host = config.get("db_host", None)
    db_edges_collection = config.get("db_edges_collection", None)

    if db_name is None or db_host is None or db_edges_collection is None:

        configs = config["task_config_files"]
        configs.append("Input.output_file=%s" % file)
        user_configs, global_config = task_helper.parseConfigs(config["task_config_files"])

        if not db_host:
            db_host = global_config["Input"]["db_host"]
        if not db_name:
            db_name = global_config["Input"]["db_name"]
        if not db_edges_collection:
            db_edges_collection = global_config["SegmentationTask"]["edges_collection"]

    print("db_host: ", db_host)
    print("db_name: ", db_name)
    print("db_edges_collection: ", db_edges_collection)
    # exit(0)
    myclient = pymongo.MongoClient(db_host)
    assert db_name in myclient.database_names(), (
        "db_name %s not found!!!" % db_name)

    return (db_name, db_host, db_edges_collection)


def load_config(config_f):

    config_f = config_f.rstrip('/')
    if config_f.endswith(".zarr"):
        config = {}
        config["file"] = config_f
        config["out_file"] = config_f
        config["raw_file"] = config_f

    else:

        with open(config_f) as f:
            config = json.load(f)

        db_name, db_host, db_edges_collection = get_db_names(config, file)
        config["db_name"] = db_name
        config["db_host"] = db_host
        config["db_edges_collection"] = db_edges_collection

    if "file" in config:
        file = config["file"]
    else:
        file = config["segment_file"]
        config["file"] = file

    for f in [
            "mask_file",
            "affs_file",
            "fragments_file",
            # "gt_file",
            "segment_file",
            ]:
        if f not in config:
            config[f] = file

    if "script_name" not in config:
        script_name = os.path.basename(config_f)
        script_name = script_name.split(".")[0]
        config["script_name"] = script_name

    script_name = config["script_name"]

    if "out_file" not in config:
        out_file = config["zarr"]["dir"] + "/" + script_name + ".zarr"
        config["out_file"] = out_file

    if 'raw_file' not in config:
        raw_file = config["zarr"]["dir"] + "/" + script_name + ".zarr"
        config["raw_file"] = raw_file

    if "mask_ds" not in config:
        config["mask_ds"] = "volumes/labels/labels_mask_z"
    if "affs_ds" not in config:
        config["affs_ds"] = "volumes/affs"
    if "fragments_ds" not in config:
        config["fragments_ds"] = "volumes/fragments"
    if "raw_ds" not in config:
        config["raw_ds"] = "volumes/raw"
    if "myelin_ds" not in config:
        config["myelin_ds"] = "volumes/myelin"
    if "segmentation_skeleton_ds" not in config:
        config["segmentation_skeleton_ds"] = "volumes/segmentation_skeleton"
    if "unlabeled_ds" not in config:
        config["unlabeled_ds"] = "volumes/labels/unlabeled_mask_skeleton"

    return config


def add_ng_layer(s, a, name, shader=None):

    if shader == 'rgb':
        shader="""void main() { emitRGB(vec3(toNormalized(getDataValue(0)), toNormalized(getDataValue(1)), toNormalized(getDataValue(2)))); }"""

    if shader == '255':
        shader="""void main() { emitGrayscale(float(getDataValue().value)); }"""

    if shader == '1':
        shader="""void main() { emitGrayscale(float(getDataValue().value)*float(255)); }"""

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


def print_ng_link(viewer):

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

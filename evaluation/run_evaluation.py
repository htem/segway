import argparse
import json
import os
from task_05_graph_evaluation_print_error import compare_segmentation_to_ground_truth_skeleton

# Consider altering task_defaults/configs to reflect actual method parameters

def parseConfigs(path):
    global_configs = {}
    # hierarchy_configs = collections.defaultdict(dict)

    # first load default configs if avail
    try:
        default_filepath = os.path.dirname(os.path.realpath(__file__))
        config_file = default_filepath + '/' + "task_defaults.json"
        with open(config_file, 'r') as f:
            global_configs = json.load(f)
    except Exception:
        print("Default task config not loaded")
        pass
    with open(path, 'r') as f:
        print("\nloading provided config %s" % path)
        new_configs = json.load(f)
        keys = set(list(global_configs.keys())).union(list(new_configs.keys()))
        for k in keys:
            if k in global_configs:
                if k in new_configs:
                    global_configs[k].update(new_configs[k])
            else:
                global_configs[k] = new_configs[k]
    print("\nconfig loaded")
    return global_configs


def construct_name_mapping(paths, names):
    d = {}
    for p, n in zip(paths, names):
        d[p] = n
    return d


def run_evaluation(
        config_path, num_processes, with_interpolation, filename):

    config = parseConfigs(config_path)
    if "skeleton" in config["Input"]:
        skeleton = config["Input"]["skeleton"]
    elif "skeleton_csv" in config["Input"]:
        skeleton = config["Input"]["skeleton_csv"]
    elif "skeleton_json" in config["Input"]:
        skeleton = config["Input"]["skeleton_json"]
    else:
        print("please provide path to skeleton or check the keyword in json \
               file")
    model_name_mapping = {}
    if "segment_names" in config["Input"]:
        model_name_mapping = construct_name_mapping(
            config["Input"]["segment_volumes"],
            config["Input"]["segment_names"])
        print(model_name_mapping)

    skeleton_configs = config["AdditionalFeatures"]
    skeleton_configs["with_interpolation"] = with_interpolation
    skeleton_configs["skeleton_path"] = skeleton
    config["Input"]["voxel_size"] = tuple(config["Input"]["voxel_size"])
    output_configs = config["Input"]
    output_configs["output_path"] = os.path.dirname(config_path)
    output_configs["config_JSON"] = filename
    output_configs["write_CSV"] = config['Output']['write_CSV']
    output_configs["skeleton_IDs_to_print"] = config['Output']['skeleton_IDs_to_print']
    parameter_configs = {}
    parameter_configs["output"] = output_configs
    parameter_configs["skeleton"] = skeleton_configs
    compare_segmentation_to_ground_truth_skeleton(
        config["Input"]["segment_dataset"],
        config["Input"]["segment_volumes"],
        model_name_mapping,
        num_processes,
        parameter_configs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="plot the graph with \
                                                  evaluation matrices \
                                                  num_split_merge_error,rand,\
                                                  voi ||or just print out the \
                                                  coordinates of split error \
                                                  and merge error")
    parser.add_argument("config", help="provide the path to configs with input \
                                        information")
    parser.add_argument(
        "-p",
        "--processes",
        help="Number of processes to use, default to 8",
        type=int,
        default=16)
    parser.add_argument("-i", "--interpolation",
                        default=False, choices=[True, False],
                        help="graph with interpolation or not")
    args = parser.parse_args()

    run_evaluation(args.config, args.processes, args.interpolation,
                   args.config.split("/")[-1].split(".")[0])

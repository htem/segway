import argparse
import json
import os

import extract_segId_from_prediction
from task_05_graph_evaluation_print_error import get_multi_merge_split_error, \
                                                 quick_compare_with_graph,\
                                                 compare_threshold_multi_model


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
        config_path, mode, num_process, with_interpolation, filename):

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

    if mode == "print":
        get_multi_merge_split_error(
            skeleton,
            config["Input"]["segment_volumes"],
            config["Input"]["segment_dataset"],
            config["PrintSplitMergeErrorTask"]["chosen_error_type"],
            num_process,
            with_interpolation,
	    config["AdditionalFeatures"]["step"],
            config["AdditionalFeatures"]["leaf_node_removal_depth"],  
            config["AdditionalFeatures"]["z_weight_multiplier"],
            config["AdditionalFeatures"]["ignore_glia"])
    elif mode == "quickplot":
        quick_compare_with_graph(
            config["Input"]["segment_dataset"],
            filename,
            skeleton,
            config["Input"]["segment_volumes"],
            model_name_mapping,
            num_process,
            os.path.dirname(config_path),
            # config["Output"]["output_path"],
            with_interpolation,
    	    config["AdditionalFeatures"]["step"],
            config["AdditionalFeatures"]["ignore_glia"],
            config["AdditionalFeatures"]["leaf_node_removal_depth"],
            config["Input"]["markers"],
            config["Input"]["colors"]
            )

    elif mode == "plot":
        compare_threshold_multi_model(
            config["Input"]["segment_dataset"],
            filename,
            skeleton,
            config["Input"]["segment_volumes"],
            config["GraphMatricesTask"]["chosen_matrices"],
            num_process,
            os.path.dirname(config_path),
            # config["Output"]["output_path"],
            with_interpolation,
            config["AdditionalFeatures"]["step"],
            config["AdditionalFeatures"]["ignore_glia"],
            config["AdditionalFeatures"]["leaf_node_removal_depth"],
            config["Input"]["markers"],
            config["Input"]["colors"])

    else:
        print("check if the mode is within plot ,quickplot, print, and check \
              the parsing code")


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
        "-m",
        "--mode",
        choices=["print", "quickplot", "plot"],
        default="quickplot",
        help="print the coordinate of errors; quickplot means you get the rand,\
              voi and split merge error directly, plot means you can choose \
              any combination of matrices")
    parser.add_argument(
        "-p",
        "--processes",
        help="Number of processes to use, default to 16",
        type=int,
        default=16)
    parser.add_argument("-i", "--interpolation",
                        default=True, choices=[True, False],
                        help="graph with interpolation or not")
    parser.add_argument("--cache_segmentation",
                        default=1, type=int,
                        help="Cache segmentation volume or not")
    # parser.add_argument("-s", "--step",
    #                     default=40,type=int,
    #                     help="step of interpolation")
    # parser.add_argument("-z", "--zMultiplier",
    #                     default=1,type=int,
    #                     help="z_weight_multiplier")
    # parser.add_argument("-f","--filename",help="name for the graph",\
    #                     default="test")
    args = parser.parse_args()

    if args.cache_segmentation is not 1:
        extract_segId_from_prediction.CACHE_SEGMENT_VOL = False

    # if len(args.config) == 0:
    #     print("Please provide configs, now running default task with config \
    #            task_defaults.json")
    # if len(args.mode) == 0:
    #     print("Please provide the mode from 'quickplot','plot','print'")
    run_evaluation(args.config, args.mode, args.processes, args.interpolation,
                   args.config.split("/")[-1].split(".")[0])

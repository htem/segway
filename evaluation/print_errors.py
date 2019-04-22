from task_05_graph_evaluation_print_error import get_merge_split_error,\
                                                 quick_compare_with_graph,\
                                                 compare_threshold_multi_model
import argparse
import json
import os


def parseConfigs(path):
    global_configs = {}

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

    model_name_mapping = {}
    if "segment_names" in config["Input"]:
        model_name_mapping = construct_name_mapping(
            config["Input"]["segment_volumes"],
            config["Input"]["segment_names"])
        print(model_name_mapping)

    if mode == "print":
        get_merge_split_error(
            config["Input"]["skeleton_csv"],
            config["Input"]["segment_volumes"],
            config["Input"]["segment_dataset"],
            config["PrintSplitMergeErrorTask"]["chosen_error_type"],
            num_process,
            with_interpolation
        )
    elif mode == "quickplot":
        quick_compare_with_graph(
            config["Input"]["segment_dataset"],
            filename,
            config["Input"]["skeleton_csv"],
            config["Input"]["segment_volumes"],
            model_name_mapping,
            num_process,
            os.path.dirname(config_path),
            # config["Output"]["output_path"],
            with_interpolation)
    elif mode == "plot":
        compare_threshold_multi_model(
            config["Input"]["segment_dataset"],
            filename,
            config["Input"]["skeleton_csv"],
            config["Input"]["segment_volumes"],
            config["GraphMatricesTask"]["chosen_matrices"],
            num_process,
            os.path.dirname(config_path),
            # config["Output"]["output_path"],
            with_interpolation)
    else:
        print("check if the mode is within plot ,quickplot, print, and check \
              the parsing code")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Print segmentation errors")
    parser.add_argument(
        "csv",
        help="Path to GT CSV")
    parser.add_argument(
        "path",
        help="Path to ZARR segmentation")
    parser.add_argument(
        "-m",
        "--mode",
        choices=["split", "merge", "both"],
        default="both",
        help="")
    args = parser.parse_args()

    # assume that volume is zarr
    path = args.path.split("zarr")
    vol = path[0] + "zarr"
    ds = path[1][1:]
    # ds = "segmentation_0.800"

    print(vol)
    print(ds)

    get_merge_split_error(
        args.csv,
        vol,
        ds,
        args.mode,
        num_process=8,
        with_interpolation=True
    )

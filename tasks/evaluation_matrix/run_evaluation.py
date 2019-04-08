from task_05_graph_evaluation_print_error import get_multi_merge_split_error, quick_compare_with_graph,compare_threshold_multi_model
import argparse
import json

def parseConfigs(path):
    global_configs = {}
    user_configs = {}
    #hierarchy_configs = collections.defaultdict(dict)

    # first load default configs if avail
    try:
        config_file = "task_defaults.json"
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

def run_evaluation(config_path, mode, num_process,with_interpolation,filename):
    config = parseConfigs(config_path)
    if mode == "print":
        get_multi_merge_split_error(
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
            num_process,
            config["Output"]["output_path"],
            with_interpolation          
        )
    elif mode == "plot":
        compare_threshold_multi_model(
            config["Input"]["segment_dataset"],
            filename,
            config["Input"]["skeleton_csv"],
            config["Input"]["segment_volumes"],
            config["GraphMatricesTask"]["chosen_matrices"],
            num_process,
            config["Output"]["output_path"],
            with_interpolation            
        )
    else:
        print("check if the mode is within plot ,quickplot, print, and check the parsing code")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="plot the graph with evaluation matrices num_split_merge_error, rand, voi  || or  just print out the coordinates of split error and merge error")
    parser.add_argument("config",help="provide the path to configs with input information")
    parser.add_argument("mode",choices = ["print","quickplot","plot"],help="print the coordinate of errors; quickplot means you get the rand, voi and split merge error directly, plot means you can choose any combination of matrices") 
    parser.add_argument("-p","--processes",help="Number of processes to use, default to 1",type=int,default=1)
    parser.add_argument("-i","--interpolation",default="True",choices = ["True","False"],help="graph with interpolation or not")
    parser.add_argument("-f","--filename",help="name for the graph",default="test")
    args = parser.parse_args()

    # if len(args.config) == 0:
    #     print("Please provide configs, now running default task with config task_defaults.json")
    # if len(args.mode) == 0:
    #     print("Please provide the mode from 'quickplot','plot','print'")
    run_evaluation(args.config, args.mode, args.processes,args.interpolation,args.filename)
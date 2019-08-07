import argparse
import json
import os
from task_05_graph_evaluation_print_error import compare_segmentation_to_ground_truth_skeleton
from extract_segId_from_prediction import load_lut, test_lut

# Consider altering task_defaults/configs to reflect actual method parameters

def parse_configs(path):
    global_configs = {}
    # hierarchy_configs = collections.defaultdict(dict)

    # first load default configs if avail
    try:
        default_filepath = os.path.dirname(os.path.realpath(__file__))
        config_file = default_filepath + '/' + 'task_defaults.json'
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


# clean this method up and/or consider reformatting the config JSONs and task_defaults
def format_parameter_configs(config):
    skeleton_configs = config['AdditionalFeatures']
    skeleton_configs['with_interpolation'] = config['AdditionalFeatures']['with_interpolation']
    skeleton_configs['skeleton_path'] = config['Input']['skeleton']
    
    error_count_configs = config['AdditionalFeatures']

    config['Input']['voxel_size'] = tuple(config['Input']['voxel_size'])
    output_configs = config['Input']
    output_configs['output_path'] = config['Output']['output_path']
    output_configs['config_JSON'] = config['file_name']
    output_configs['write_CSV'] = config['Output']['write_CSV']
    output_configs['write_TXT'] = config['Output']['write_TXT']
    output_configs['skeleton_IDs_to_print'] = config['Output']['skeleton_IDs_to_print']
    
    parameter_configs = {}
    parameter_configs['skeleton'] = skeleton_configs
    parameter_configs['error_count'] = error_count_configs
    parameter_configs['output'] = output_configs
    return parameter_configs


def run_evaluation(
        config_path, num_processes, file_name):
    config = parse_configs(config_path)
    config['Output']['output_path'] = os.path.dirname(config_path)
    if 'skeleton_csv' in config['Input']:
        config['Input']['skeleton'] = config['Input']['skeleton_csv']
    elif 'skeleton_json' in config['Input']:
        config['Input']['skeleton'] = config['Input']['skeleton_json']
    elif 'skeleton' not in config['Input']:
        print("Please provide path to skeleton or check the keyword in json \
               file")
    model_name_mapping = {}
    if 'segment_names' in config['Input']:
        model_name_mapping = construct_name_mapping(
            config['Input']['segment_volumes'],
            config['Input']['segment_names'])
        print(model_name_mapping)
    config['file_name'] = file_name
    parameter_configs = format_parameter_configs(config)
    compare_segmentation_to_ground_truth_skeleton(
        config['Input']['segment_dataset'],
        config['Input']['segment_volumes'],
        model_name_mapping,
        num_processes,
        parameter_configs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='provide the path to configs with input \
                                        information')
    parser.add_argument(
        '-p',
        '--processes',
        help='Number of processes to use, default to 8',
        type=int,
        default=16)
    args = parser.parse_args()
    file_name = args.config.split('/')[-1].split('.')[0]
    run_evaluation(args.config, args.processes, file_name)

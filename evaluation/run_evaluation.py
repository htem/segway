import argparse
import json
import os
from task_05_graph_evaluation_print_error import compare_segmentation_to_ground_truth_skeleton, generate_error_plot
from operator import add
# Consider altering task_defaults/configs to reflect actual method parameters

def parse_configs(path):
    global_configs = {}
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
def format_parameter_configs(config, volume, iteration):
    
    skeleton_configs = config['AdditionalFeatures']
    skeleton_configs['skeleton_path'] = volume['skeleton'] 
    error_count_configs = config['AdditionalFeatures']
    output_configs = config['Output']
    output_configs['config_JSON'] = config['file_name']
    output_configs['voxel_size'] = tuple(config['Input']['voxel_size'])
    volume_name = get_vol_name(volume,iteration)
    

    return {'skeleton': skeleton_configs, 'error_count': error_count_configs, 'output': output_configs, 'name': volume_name}

def get_vol_name(vol, i):
    if "volume_name" not in vol:
        return "{}".format(i)
    else:
        return vol["volume_name"]

def get_weight(vol):
    if "weight" not in vol:
        return (1)
    else:
        return (vol["weight"])

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
            volume['segment_volumes'],
            config['Input']['segment_names'])
        print(model_name_mapping, "&&&&&")
    config['file_name'] = file_name

    if 'Inputs' in config:
        splits_and_merges=[]
        weights=[]
        for num, volume in enumerate(config['Inputs']):
            weights.append(get_weight(volume))
            parameter_configs = format_parameter_configs(config, volume, num)
            
            splits_and_merges.append(
                compare_segmentation_to_ground_truth_skeleton(
                config['Input']['segment_dataset'],
                volume['segment_volumes'],
                model_name_mapping,
                num_processes,
                parameter_configs))
 

        print(splits_and_merges)
        
        splits_and_merges = [format_splits_and_merges(x,weights) for x in zip(*splits_and_merges) ]
        print(splits_and_merges)
        
        splits_and_merges= add_weights(splits_and_merges, weights)

        print(splits_and_merges)

        generate_error_plot(config['Input']['segment_dataset'],config['file_name'],'Combined',
            config['Output']['output_path'],config['Output']['markers'],
            config['Output']['colors'],'number', *splits_and_merges)
    else:
        parameter_configs = format_parameter_configs(config,config['Input'],0)

        compare_segmentation_to_ground_truth_skeleton(
            config['Input']['segment_dataset'],
            config['Input']['segment_volumes'],  
            model_name_mapping,
            num_processes,
            parameter_configs)

def add_weights(splits_and_merges,weights):
    pos = -1
    weighted_list=[]
    for element in splits_and_merges:
        if isinstance(element, str):
            print(pos)
            weighted_list.append(element)
            pos += 1
        else:
            
            new_list =[ i*weights[pos] for i in element]
            weighted_list.append(new_list)
    return weighted_list
            


def format_splits_and_merges(list_of_iterables, weights):
    
    if isinstance(list_of_iterables[1], list):
        unweighted_list= (list(sum(x) for x in zip(*list_of_iterables))) 
        return unweighted_list
    
    elif isinstance(list_of_iterables[1], str):
        if (all(x == list_of_iterables[0] for x in list_of_iterables)):
            return(list_of_iterables[0])
        else: return(list_of_iterables)

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

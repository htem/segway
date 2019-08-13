import json
import os
import sys
import time
import argparse
import daisy
import numpy as np

from synful import detection, database
import synful.synapse

def parse_configs():
    parser = argparse.ArgumentParser()
    parser.add_argument('user_path')
    args = parser.parse_args()
    global_configs = {}
    
    default_config_directory = os.path.dirname(os.path.realpath(__file__))
    default_config_path = os.path.join(default_config_directory,
                                       'synapse_task_defaults.json')
    with open(default_config_path, 'r') as f:
        global_configs = json.load(f)

    with open(args.user_path, 'r') as f:
        user_configs = json.load(f)
    
    for key in global_configs:
        try:
            global_configs[key].update(user_configs[key])
        except KeyError:
            pass
    
    print("Config loaded")
    return global_configs


def load_synapses_from_catmaid_json(json_path):
    with open(json_path, 'r') as f:
        catmaid_data = json.load(f)
    synapses = {}
    for skeleton_id, sk_dict in catmaid_data['skeletons'].items():
        for connector, connector_dict in sk_dict['connectors'].items():
            if connector_dict['presnaptic_to'] == []:
                presynaptic = [skeleton_id]
                postsynaptic = connector_dict['postsynaptic_to']
            else:
                presynaptic = connector_dict['presnaptic_to']
                postsynaptic = [skeleton_id]
            zyx_coord = (int(connector_dict['location'][2]),
                         int(connector_dict['location'][1]),
                         int(connector_dict['location'][0]))
            synapses[connector] = {'skeleton_id': skeleton_id,
                                   'presynaptic': presynaptic,
                                   'postsynaptic': postsynaptic,
                                   'zyx_coord': zyx_coord}
    return synapses
            

def load_predictions_from_zarr(dataset, predictions, ground_truth):
    synapse_predictions = daisy.open_ds(dataset, predictions)
    synapse_predictions = synapse_predictions[synapse_predictions.roi]
    print(synapse_predictions.roi)
    synapses_outside_roi = []
    for synapse, attr in ground_truth.items():
        try:
            attr['score'] = synapse_predictions[daisy.Coordinate(attr['zyx_coord'])]
        except AssertionError:
            synapses_outside_roi.append(synapse)
    for synapse in synapses_outside_roi:
        ground_truth.pop(synapse)
    return ground_truth

if __name__ == '__main__':
    configs = parse_configs()
    print(configs)
    synapses = load_synapses_from_catmaid_json(configs['Input']['skeleton'])
    print(len(synapses))
    
    load_predictions_from_zarr(configs['Input']['dataset'],
                               configs['Input']['predictions'][0],
                               synapses)
    
    true_positives = {synapse for synapse in synapses 
                      if synapses[synapse]['score'] != 0}
    print("%s / %s synapses were true positives" %
                (len(true_positives), len(synapses)))
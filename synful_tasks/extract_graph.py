import numpy as np
import networkx as nx
from database_synapses import SynapseDatabase
from database_superfragments import SuperFragmentDatabase
import os
import sys
sys.path.insert(0, '/n/groups/htem/temcagt/datasets/cb2/segmentation/tri/repos/funlib.show.neuroglancer')
sys.path.insert(0, '/n/groups/htem/Segmentation/tmn7/segwaytool.proofreading')
import json
import segwaytool.proofreading
import segwaytool.proofreading.neuron_db_server
import time
import matplotlib.pyplot as plt
import itertools
from collections import defaultdict
import scipy.sparse
import csv
import pandas as pd
from daisy import Coordinate


def plot_adj_mat(A, configs):

    """ Plot Adj matrix according to the configs
    configs is a dictionary with the following keys
    ['full', 'some', 'pre', 'post', 'threshold_value'(int/float)] 
    and the output paths as values

    configs values will include the list of neuorns of interest
    """

    # if the threshold is present, all the plots will be done considering that threshold
    if 'threshold' in configs.keys():
        A[A<=configs['threshold']] = 0

    if 'full' in configs.keys():
        fig = plt.figure(figsize=(16,15))
        ax = fig.add_subplot(111)
        i = ax.imshow(A)
        ax.set_xticks(np.arange(len(A)))
        ax.set_xticklabels(configs['full_list'], rotation=75)
        ax.set_yticks(np.arange(len(A)))
        ax.set_yticklabels(configs['full_list'])
        plt.colorbar(i, ax=ax)
        fig.savefig(configs['full'])

    if 'some' in configs.keys():
        fig = plt.figure(figsize=(16,15))
        ax = fig.add_subplot(111)
        i = ax.imshow(adj_configs['small_adj'])
        ax.set_xticks(np.arange(len(adj_configs['small_adj'])))
        ax.set_xticklabels(configs['small_list'], rotation=75)
        ax.set_yticks(np.arange(len(adj_configs['small_adj'])))
        ax.set_yticklabels(configs['small_list'])
        plt.colorbar(i, ax=ax)
        fig.savefig(configs['some'])

    if 'pre' in configs.keys():
        # shrink the adjency matrix
        mat = A[:,[configs['full_list'].index(i) for i in configs['list_posts']]]

        fig = plt.figure(figsize=(16,15))
        ax = fig.add_subplot(111)
        i = ax.imshow(mat)
        ax.set_xticks(np.arange(mat.shape[1]))
        ax.set_xticklabels(configs['list_posts'], rotation=75)
        ax.set_yticks(np.arange(mat.shape[0]))
        ax.set_yticklabels(configs['full_list'])
        plt.colorbar(i, ax=ax)
        fig.savefig(configs['pre'])

    if 'post' in configs.keys():
        # shrink the adjency matrix
        mat = A[[configs['full_list'].index(i) for i in configs['list_pres']],:]

        fig = plt.figure(figsize=(16,15))
        ax = fig.add_subplot(111)
        i = ax.imshow(mat)
        ax.set_xticks(np.arange(mat.shape[1]))
        ax.set_xticklabels(configs['full_list'], rotation=75)
        ax.set_yticks(np.arange(mat.shape[0]))
        ax.set_yticklabels(configs['list_pres'])
        plt.colorbar(i, ax=ax)
        fig.savefig(configs['post'])

    return

def create_edges_graph(G, edge_list, weights=[]):

    if len(weights) == 0:
        G.add_edges_from(edge_list)
        print("## Info : Edges created!")
    else:

        for i in range(len(edge_list)):
            G.add_edge(edge_list[i][0], edge_list[i][1], weight=weights[i])

        print("## Info : Edges and weights created!")

    return G

def save_edges(edge_list, 
               weights, 
               synapses_locs, 
               output_edges):

    columns = ['pre_partner', 'post_partner', 'weight', 'synapses_locs']
    edge_list = np.array(edge_list)
    df = pd.DataFrame(list(zip(edge_list[:,0],edge_list[:,1],weights, synapses_locs)),
                      columns=columns)

    df.to_csv(output_edges)

    return df

def compute_weights(edge_list,
                    synapse_list, 
                    syn_db,
                    g,
                    score_threshold,
                    sf_to_neurons,
                    voxel_size_xyz,
                    mode='count',
                    dist=False):

    """
    Compute weights according to the area of the synapse
    if area is True the area of the synapses is taken into consideration
    if dist is True, the distance from the soma is also considered
    if area and dist are False the weights will be the number of synapses
    
    mode = "count"/"area"
    
    """

    mode_area = mode=="area"

    print("## Info : Computing the weights of the graph...")
    start = time.time()

    syn_weights = defaultdict(float)
    # syn_weights = dict()

    # get synapse attributes
    score_threshold = 0.60  # TO FIX with config file
    query = { '$and' : [{'id' : { '$in' : list(synapse_list)}}, 
                                {'score': { '$gt': score_threshold }} ]}

    #query = {'id' : { '$in' : list(synapse_list) }}
    synapses_query = syn_db.synapses.find(query)
    synapses_dict = defaultdict(list)


    for syn in synapses_query:
        # print(syn)
        pre_neuron = syn['id_superfrag_pre']
        post_neuron = syn['id_superfrag_post']
        if pre_neuron not in sf_to_neurons or post_neuron not in sf_to_neurons:
            continue

        pre_neuron = sf_to_neurons[pre_neuron]
        post_neuron = sf_to_neurons[post_neuron]

        if pre_neuron == post_neuron:
            continue

        weight = 1
        if mode_area:
            weight = syn['area']/1e+3
        if dist:
            soma_loc = np.array([g.nodes[post_neuron]['x'],
                                 g.nodes[post_neuron]['y'],
                                 g.nodes[post_neuron]['z']])

            syn_loc = np.array([syn['x'],syn['y'],syn['z']])
            distance = np.linalg.norm(soma_loc-syn_loc)
            weight = weight/distance

        syn_weights[(pre_neuron, post_neuron)] += weight
        synapses_dict[(pre_neuron, post_neuron)].append([int(syn['x']/voxel_size_xyz[0]),
                                                         int(syn['y']/voxel_size_xyz[1]),
                                                         int(syn['z']/voxel_size_xyz[2])])

    weights = []
    synapses_locs = []
    filt_edge_list = edge_list.copy()  

    for e in edge_list:
        if e in syn_weights:
            weights.append(syn_weights[e])
            synapses_locs.append(synapses_dict[e])

        elif e not in syn_weights:
            print("Edge %s not found in synapse attributes" % str(e))
            filt_edge_list.remove(e)
            #continue
        # assert e in syn_weights
        #weights.append(syn_weights[e])

    print("Weights creation took %f s" % (time.time()-start))

    return weights, filt_edge_list, synapses_locs

def create_edges_list(neurons_dict_sf, sf_db, sf_to_neurons):

    neuron_list = np.array(list(neurons_dict_sf.keys()))  # all neurons
    edge_list = set()
    synapse_list = set()

    for nid in neuron_list:
        # for each neuron, we get their post partners as sf
        # convert them to neuron_id and add directed edge

        sfs_dict = _get_superfragments_info_db(neurons_dict_sf[nid],
                                                sf_db)      
        for sf in sfs_dict:

            post_partners_sf = sf['post_partners']
            # print("post_partners_sf:", post_partners_sf)
            for post_sf in post_partners_sf:
                if post_sf not in sf_to_neurons:
                    # post neuron not in input list
                    continue
                post_neuron = sf_to_neurons[post_sf]
                if post_neuron != nid:
                    edge_list.add((nid, post_neuron))

            pre_partners_sf = sf['pre_partners']
            # print("pre_partners_sf:", pre_partners_sf)
            for pre_sf in pre_partners_sf:
                if pre_sf not in sf_to_neurons:
                    # post neuron not in input list
                    continue
                pre_neuron = sf_to_neurons[pre_sf]
                if pre_neuron != nid:
                    edge_list.add((pre_neuron, nid))

            synapse_list.update(sf['syn_ids'])

    edge_list = list(edge_list)
    return edge_list, synapse_list

def create_neurons_dict_sf(neurons_list,neuron_db):
    # dictionary with neurons as keys and their sf as values
    neurons_dict_sf= dict()

    for nid in neurons_list:
        superfragments = neuron_db.get_neuron(nid).to_json()['segments']
        sfs_list = [int(item) for item in superfragments]   
        neurons_dict_sf[nid] = sfs_list

    return neurons_dict_sf

def create_sf_dict_neurons(neurons_list,neurons_dict_sf):
    # create reverse dictionary 
    sf_to_neurons = dict()
    for nid in neurons_list:
        for sfid in neurons_dict_sf[nid]:
            sf_to_neurons[sfid] = nid

    return sf_to_neurons

def create_nodes_graph(neurons_list, nodes_attr):

    G = nx.DiGraph()
    for i, n in enumerate(neurons_list):
        G.add_node(n)

    nx.set_node_attributes(G, nodes_attr)
    print("### Info : Number of nodes in the graph : ", G.number_of_nodes())

    return G

def _get_superfragments_info_db(superfragments_list,
                                sf_db):

    super_fragments_file = "/n/pure/htem/Segmentation/cb2_v4/output.zarr"
    super_fragments_dataset = "volumes/super_1x2x2_segmentation_0.500_mipmap/s4"
    sfs_dict = sf_db.read_superfragments(sf_ids=superfragments_list)
    
    return sfs_dict

def _get_neurons_info_db(neurons_list, neuron_db):

    nodes_attr = {}
    for nid in neurons_list:
        neuron = neuron_db.get_neuron(nid).to_json()
        # create dictionary with attributes per neuron
        idict = dict()
        idict['cell_type'] = neuron['cell_type']
        idict['x'] =  neuron['soma_loc']['x']
        idict['y'] =  neuron['soma_loc']['y']
        idict['z'] =  neuron['soma_loc']['z']
        idict['tags'] = neuron['tags']
        idict['finished'] = neuron['finished']
        # assign attributes dictionary 
        nodes_attr[nid] = idict

    return nodes_attr

def _connect_DBs(db_host,
                 db_name,
                 db_name_n):

    syn_db = SynapseDatabase(db_name= db_name, db_host= db_host,
    db_col_name='synapses', )

    sf_db = SuperFragmentDatabase(
        db_name= db_name,
        db_host= db_host,
        db_col_name='superfragments',
        )

    neuron_db = segwaytool.proofreading.neuron_db_server.NeuronDBServer(
                db_name= db_name_n,
                host= db_host,
                )
    neuron_db.connect()

    return syn_db, sf_db, neuron_db


if __name__ == "__main__":

    # variables from configuration file (DB, input and output variables)

    assert len(sys.argv) == 2
    config_file = sys.argv[1]

    # open config file and create the variables 
    with open(config_file) as f:
        params = json.load(f)
        for key, item in params.items():
            vars()[key] = item

    if input_method == 'user_list':
        neurons_list = sorted(list(set(input_neurons_list)))
    elif input_method == 'all':
        # WARNING : in 'neuron_db_server.py' there is the limit of 100 neurons
        # so only 100 neurons will be queried 
        neuron_list = neuron_db.find_neuron({})
    elif input_method == 'roi':
        # query neurons if roi.contains(Coordinate(soma_loc))
        # TO IMPLEMENT ...
        pass

    # access the database and generate graph characteristics if it was not
    # already existing or if it was existing but overwrite option is True 
    if not os.path.exists(output_graph) or (os.path.exists(output_graph) and overwrite==1) :
        # assuming that if there is no output_graph there is no edge_list and 
        # adjacency matrix saved either

        # access the DB
        syn_db, sf_db, neuron_db = _connect_DBs(db_host,
                                                db_name,
                                                db_name_n
                                                )
        
        nodes_attr = _get_neurons_info_db(neurons_list, neuron_db)
        g = create_nodes_graph(neurons_list,nodes_attr)

        # create useful dictionaries:
        neurons_dict_sf = create_neurons_dict_sf(neurons_list,neuron_db)
        sf_to_neurons = create_sf_dict_neurons(neurons_list,neurons_dict_sf)



        print("### Info: Running create_edges_list...")
        edge_list, synapse_list = create_edges_list(neurons_dict_sf,
                                                    sf_db,
                                                    sf_to_neurons,
                                                    )
        print("### Info: len(edge_list) NOT filtered:", len(edge_list))
        
        # mode = 'count'
        # mode = 'area' 
        compute_distance = bool(weights_with_dist)
        
        # if user specified edges to add:
        if len(add_edge_list) > 0:
            for e in add_edge_list: 
                edge_list.append(e) 

            edge_list = list(set(edge_list))
            print("### Info: Added edges specified by the user, len(edge_list) :", len(edge_list))    

        weights, edge_list, syns_locs = compute_weights(edge_list,
                                                        synapse_list, 
                                                        syn_db,
                                                        g,
                                                        syn_score_threshold,
                                                        sf_to_neurons,
                                                        voxel_size_xyz,
                                                        mode=mode_weights, # mode
                                                        dist=compute_distance,
                                                        )

        #print("weights: ", weights)
        print(" ### Info: len(weights) (filtered edge_list): ", len(edge_list))





        # save FILE edges
        edge_list_df = save_edges(edge_list, 
                                  weights, 
                                  syns_locs, 
                                  output_edges)

        g = create_edges_graph(g, edge_list, weights)
        # save graph
        nx.write_gpickle(g, output_graph)
        
    else:
        # load graph, adj and edge list
        g = nx.read_gpickle(output_graph)
        print("### Info: Graph loaded")
        print("Number of nodes: ", g.number_of_nodes())
        edge_list_df = pd.read_csv(output_edges, index_col=0) # with info on the weights and synapses
        # edge list names
        edge_list = list(zip(*map(edge_list_df.get, ['pre_partner', 'post_partner'])))
        print("### Info: edge_list loaded")

    # if preprocessed graph is not existing or overwriting is on
    if not os.path.exists(output_graph_pp) or (os.path.exists(output_graph_pp) and overwrite==1) :

        # Preprocessing the graph, given specifications in the config file
        pre_proc = len(exclude_neurons)>0 or len(tags_to_exclude)>0 or len(exclude_edges)>0
        if pre_proc:
            
            g.remove_nodes_from(exclude_neurons)    
            filtered_nodes = []
            for tte in tags_to_exclude:
                filtered_nodes.extend([n for n,d in g.nodes(data=True) if d['cell_type'] == tte[0] 
                                      and tte[1] in d['tags']])

            g.remove_nodes_from(filtered_nodes)

            print("Num of nodes (filtered): ", g.number_of_nodes())

            g.remove_edges_from(exclude_edges)

        # rename nodes 
        # NOTE : finished tag will be added only to interneuorns with no cell type specified,
        # if instead the cell type is specified (eg basket) the rename will be basket_ and it
        # assumes the interneuron finished

        if len(rename_rules)>0:
            rename_dict = dict()

            for rule in rename_rules:
                # rule to query the node of interest
                query = rule[2]
                if len(query) == 0 :
                    # direct rename
                    rename_dict[rule[0]] = rule[1]
                elif len(query) == 1:
                    # cell type is specifies (example interneuron_ -> basket_)
                    queried_nodes = [n for n,d in g.nodes(data=True) if d[list(query.keys())[0]] == list(query.values())[0]]
                else:
                    # e.g. specified cell type and finished tag
                    nodes_dict = dict()
                    for (n,d) in g.nodes(data=True):
                        nodes_dict[n] = 0
                        for k, v in query.items():
                            if d[k] == v:
                                nodes_dict[n] += 1

                    queried_nodes = list(dict(filter(lambda val: val[1] == 2, nodes_dict.items())).keys())

                for node in queried_nodes:

                    if node.find(rule[0]) == 0 :
                        rename_dict[node] = node.replace(node[:len(rule[0])], rule[1], 1)

            g = nx.relabel_nodes(g,rename_dict)
            
        nx.write_gpickle(g, output_graph_pp)

    else:

        g = nx.read_gpickle(output_graph_pp)
        print("Num of nodes (filtered): ", g.number_of_nodes())        

    # debug specified edges: print them on file
    if debug_edges == 1: 
        deb_edge_list = pd.DataFrame()
        for i in range(len(debug_edges_list)):
            q = edge_list_df[(edge_list_df['pre_partner'] == debug_edges_list[i][0]) & 
                             (edge_list_df['post_partner'] == debug_edges_list[i][1])]
            deb_edge_list = deb_edge_list.append(q,ignore_index=True)

        deb_edge_list.to_csv(output_debug_edges)
    
    # Adjacency matrix
    A = nx.to_numpy_matrix(g)

    # ['full', 'some', 'pre_list', 'post_list']
    adj_configs = dict()
    # full list of neurons is usefull in all the cases (out of if condition)
    adj_configs['full_list'] = list(g.nodes())
    if adj_plot_thresh == 1:
        adj_configs['threshold'] = weights_threshold
    if adj_plot_all == 1:
        adj_configs['full'] = adj_all
    if adj_plot_pre == 1:
        # all the pre syn partners -> list of posts (column chunk of A)
        adj_configs['pre'] = adj_pre
        adj_configs['list_posts'] = list_posts
    if adj_plot_post == 1:
        # all the post syn partners -> list of pres (row chunk of A)
        adj_configs['post'] = adj_post     
        adj_configs['list_pres'] = list_pres
    if adj_plot_some == 1:   
        adj_configs['some'] = adj_small_list
        adj_configs['small_list'] = adj_plot_list
        adj_configs['small_adj'] = nx.to_numpy_matrix(g, nodelist=adj_plot_list)

    plot_adj_mat(A, adj_configs)
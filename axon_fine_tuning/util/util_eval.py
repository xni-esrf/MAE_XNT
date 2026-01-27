
import numpy as np
import networkx as nx
import math
from itertools import product, combinations
from util.util_inference import run_inference
from util.run_length import expected_run_length, get_skeleton_lengths
from collections import defaultdict
import csv
import os
import h5py

# Most of the functions here are modification of the evaluation functions provided in the challenge github : https://github.com/trivoldus28/xray-challenge-eval/tree/master/eval

def generate_graphs_with_seg_labels(segment_array, skeleton_path, voxel_size, offset):
    # Function to add predicted segmentation to the ground-truth graph. Information is added on the node level while ground-truth information is contained on the edge level.
    gt_graph = np.load(skeleton_path, allow_pickle=True)

    next_highest_seg_label = int(np.max(segment_array)) + 1
    nodes_outside_roi = []
    for i, (treenode, attr) in enumerate(gt_graph.nodes(data=True)):
        pos = attr["position"]
        attr['zyx_coord'] = (pos[2], pos[1], pos[0])
        # Convert from real coordinates to voxel
        attr['pix_pos'] = tuple([int(p/voxel_size) - offset[i] for i, p in enumerate(attr['zyx_coord'])])

        try:
            if np.all(np.array(list(attr['pix_pos'])) > 0):
                attr['seg_label'] = segment_array[attr['pix_pos']]
            else:
                raise IndexError
        except IndexError as e:
            nodes_outside_roi.append(treenode)
            continue
        if attr['seg_label'] == 0:

            # We'll need to relabel them to be a unique non-zero value
            # for the Rand/VOI function to work. We also count contiguous skeletal
            # nodes predicted to be 0 as split errors.
            attr['seg_label'] = next_highest_seg_label
            segment_array[attr['pix_pos']] = next_highest_seg_label
            next_highest_seg_label += 1

    for node in nodes_outside_roi:
        gt_graph.remove_node(node)

    # reassign `skeleton_id` attribute used in eval functions, because removing nodes can have split axons
    skel_clusters = nx.connected_components(gt_graph)
    for i, cluster in enumerate(skel_clusters):
        for node in cluster:
            gt_graph.nodes[node]['skeleton_id'] = i
    return gt_graph



def eval_erl(graph):
    # Evaluation of Expected run length
    node_seg_lut = {}
    for node, attr in graph.nodes(data=True):
        node_seg_lut[node] = attr['seg_label']

    # get total skel length
    skeleton_lengths = get_skeleton_lengths(
        skeletons=graph,
        skeleton_position_attributes=['zyx_coord'],
        skeleton_id_attribute='skeleton_id')
    skeleton_lengths = [l for _, l in skeleton_lengths.items() if l > 0]
    average_skel_length = np.mean(skeleton_lengths)

    erl = expected_run_length(
        skeletons=graph, skeleton_id_attribute='skeleton_id',
        node_segment_lut=node_seg_lut, skeleton_position_attributes=['zyx_coord'],
        return_merge_split_stats=False, edge_length_attribute='edge_length')
    erl_norm = erl/average_skel_length

    return erl, erl_norm


def gt_graph_into_array(test_array, gt_graph):
    # Ground truth graph into array conversion for rand indices computation
    gt_array = np.zeros_like(test_array).astype(np.uint32)
    for neuron_id, cluster in enumerate(nx.connected_components(gt_graph)): # Using nx connected components makes sure we use gt skeleton segmentation (connected with edges) and not the modified gt graph.
        for node in cluster:
            gt_array[gt_graph.nodes[node]['pix_pos']] = neuron_id
    return gt_array


def find_splits(gt_graph):
    split_errors = []
    for edge in gt_graph.edges():
        # Compare labels between connected nodes (edges correspond to the ground-truth)
        if gt_graph.nodes[edge[0]]['seg_label'] != gt_graph.nodes[edge[1]]['seg_label']:
            split_errors.append(edge)
    return split_errors


def build_segment_label_subgraph(segment_nodes, graph):
    # Build a subgraph corresponding 
    subgraph = graph.subgraph(segment_nodes)
    skeleton_clusters = nx.connected_components(subgraph)
    seg_graph = nx.Graph()
    seg_graph.add_nodes_from(subgraph.nodes)
    seg_graph.add_edges_from(subgraph.edges)
    for skeleton_1, skeleton_2 in combinations(skeleton_clusters, 2):
        try:
            node_1 = skeleton_1.pop()
            node_2 = skeleton_2.pop()
            if graph.nodes[node_1]['skeleton_id'] == graph.nodes[node_2]['skeleton_id']:
                seg_graph.add_edge(node_1, node_2)
        except KeyError:
            pass
    return seg_graph


def get_closest_node_pair_between_two_skeletons(skel1, skel2, graph):
    # Returns the closest pair of nodes on 2 skeletons
    shortest_len = math.inf
    for node1, node2 in product(skel1, skel2):
        coord1, coord2 = graph.nodes[node1]['zyx_coord'], graph.nodes[node2]['zyx_coord']
        distance = math.sqrt(sum([(a-b)**2 for a, b in zip(coord1, coord2)]))
        if distance < shortest_len:
            shortest_len = distance
            edge_attributes = {'distance': shortest_len}
            closest_pair = (node1, node2, edge_attributes)
    return closest_pair


def find_merge_errors(graph):
    # Counts the number of merges in the segmentation
    seg_dict = {}
    for nid, attr in graph.nodes(data=True):
        seg_label = attr['seg_label']
        assert seg_label != 0, "Processed predicted labels cannot be 0"
        try:
            seg_dict[seg_label].add(nid)
        except KeyError:
            seg_dict[seg_label] = {nid}

    merge_errors = set()
    for seg_label, nodes in seg_dict.items():
        seg_graph = build_segment_label_subgraph(nodes, graph)
        skel_clusters = list(nx.connected_components(seg_graph))
        if len(skel_clusters) <= 1:
            continue
        potential_merge_sites = []
        for skeleton_1, skeleton_2 in combinations(skel_clusters, 2):
            shortest_connection = get_closest_node_pair_between_two_skeletons(
                                  skeleton_1, skeleton_2, graph)
            potential_merge_sites.append(shortest_connection)

        merge_sites = [(error_site[0], error_site[1]) for error_site in potential_merge_sites]
        merge_errors |= set(merge_sites)

    return merge_errors


def rand_indices(gt_array, pred_array):
    # Rand indices computation as described in "Objective Criteria for the Evaluation of Clustering Methods"
    co_occ = defaultdict(lambda: defaultdict(int))
    occ_gt = defaultdict(int)
    occ_pred = defaultdict(int)
    gt_array = np.ravel(gt_array, order='A')
    pred_array = np.ravel(pred_array, order='A')
    total_labels = 0
    for i in range(gt_array.shape[0]):
        # Compare on points annotated in the ground-truth skeleton 
        if gt_array[i] > 0:
            total_labels += 1
            label_gt = gt_array[i]
            label_pred = pred_array[i]
            co_occ[label_gt][label_pred] += 1
            occ_gt[label_gt] += 1
            occ_pred[label_pred] += 1
    
    sum_co_occ = 0
    for _ in co_occ.values():
        for num_el in _.values():
            sum_co_occ += num_el**2
    
    sum_gt = 0
    for num_el in occ_gt.values():
        sum_gt += num_el**2
    
    sum_pred = 0
    for num_el in occ_pred.values():
        sum_pred += num_el**2
    return sum_co_occ/sum_gt, sum_co_occ/sum_pred

    
def run_eval(offset, volume_size, agreg, file, model, model_folder, rootdata, batch_size, threshold):
    # Evaluation pipeline from affinity prediction to evaluation of the predicted segmentation
    size_voxel = 33

    if file == 'train':
        skeleton_path = os.path.join(rootdata,'XPRESS_training_skels.npz')
    if file == 'val':
        skeleton_path = os.path.join(rootdata, 'XPRESS_validation_skels.npz')


    save_folder =  f'result_{file}_{offset}_{volume_size}_{agreg}_{threshold}/'

    segmentation = run_inference(file, model, model_folder, rootdata, batch_size, agreg, volume_size, offset, threshold)

    gt_graph = generate_graphs_with_seg_labels(segmentation, skeleton_path, size_voxel, offset)
    gt_array = gt_graph_into_array(segmentation, gt_graph)
    # Total number of axons in the ground truth skeleton (cropped in the region of interest)
    n_neurons = len(list(nx.connected_components(gt_graph)))

    splits = find_splits(gt_graph)
    merges = find_merge_errors(gt_graph)
    erl, erl_norm = eval_erl(gt_graph)
    rand_split, rand_merge = rand_indices(gt_array, segmentation)
    os.makedirs(os.path.join(model_folder,save_folder), exist_ok=True)
    with open(os.path.join(model_folder,save_folder,'eval.txt'), 'w') as f:
        f.write(f'num neurons : {n_neurons}\n')
        f.write(f'num splits : {len(splits)}\n')
        f.write(f'num merges : {len(merges)}\n')
        f.write(f'n split per neuron : {len(splits)/n_neurons}\n')
        f.write(f'n merges per neuron : {len(merges)/n_neurons}\n')
        f.write(f'ERL : {erl}\n')
        f.write(f'ERL norm : {erl_norm}\n')
        f.write(f'rand split : {rand_split}\n')
        f.write(f'rand merge : {rand_merge}\n')
        f.write(f'xpress score : {((rand_merge+rand_split)/2 + erl_norm)/2}')

    print('num neurons :' ,n_neurons)
    print('num splits : ', len(splits))
    print('num merges : ', len(merges))
    print('n split per neuron :', len(splits)/n_neurons)
    print('n merges per neuron :', len(merges)/n_neurons)
    print(f'ERL : {erl}')
    print(f'ERL norm : {erl_norm}')
    print(f'rand split : {rand_split}')
    print(f'rand merge : {rand_merge}')
    print(f'XPRESS score : {((rand_merge+rand_split)/2 + erl_norm)/2}')


def validate_model(model, model_folder, rootdata, batch_size, num_epoch, thresholds, whole=False):
    # Evalution on a list of thresholds used for model hyperparameter tuning, write the results in a csv file
    file_save = os.path.join(model_folder, 'res.csv')

    size_voxel = 33
    agreg = 'hann'
    file = 'val'

    skeleton_path = os.path.join(rootdata, 'XPRESS_validation_skels.npz')
    if whole :
        volume_size = (699, 699, 699)
    else:
        volume_size = (174, 699, 699)
    offset = (252,252,252)

    list_xpress = []
    # list_rand = []
    if len(thresholds) > 1:
        affs = run_inference(file, model, model_folder, rootdata, batch_size, agreg, volume_size, offset, threshold=0)
        file_aff = f'result_{file}_{offset}_{volume_size}/'
        os.makedirs(os.path.join(model_folder,file_aff), exist_ok=True)
        # Save affinity map prediction
        with h5py.File(os.path.join(model_folder,file_aff,'affs.h5'), 'w') as f:
            f.create_dataset("volumes", data=affs)
        f.close()
    for threshold in thresholds:
        segmentation = run_inference(file, model, model_folder, rootdata, batch_size, agreg, volume_size, offset, threshold=threshold)

        gt_graph = generate_graphs_with_seg_labels(segmentation, skeleton_path, size_voxel, offset)
        gt_array = gt_graph_into_array(segmentation, gt_graph)
        erl, erl_norm = eval_erl(gt_graph)
        rand_split, rand_merge = rand_indices(gt_array, segmentation)
        list_xpress.append(((rand_merge+rand_split)/2 + erl_norm)/2)
        # list_rand.append((rand_merge+rand_split)/2)

    with open(file_save, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow([num_epoch] + list_xpress)


def test_model(model, model_folder, rootdata, batch_size, threshold):
    # Function to perform evaluation on six subdivision of the validation set "independant" from the one used during training
    file_save = os.path.join(model_folder, 'test.csv')

    size_voxel = 33
    agreg = 'hann'
    file = 'val'

    skeleton_path = os.path.join(rootdata, 'XPRESS_validation_skels.npz')
    volume_size = (174, 699, 349)
    # List of offsets on which we evaluate
    offsets = [(426,252,252), (426,252,601), (600,252,252), (600,252,601), (774,252,252), (774,252,601)]

    with open(file_save, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['offsets :'] + offsets)    
    
    list_xpress = []
    rand_splits = []
    rand_merges = []
    erls = []
    for offset in offsets:
        segmentation = run_inference(file, model, model_folder, rootdata, batch_size, agreg, volume_size, offset, threshold=threshold)

        gt_graph = generate_graphs_with_seg_labels(segmentation, skeleton_path, size_voxel, offset)
        gt_array = gt_graph_into_array(segmentation, gt_graph)
        erl, erl_norm = eval_erl(gt_graph)
        rand_split, rand_merge = rand_indices(gt_array, segmentation)
        rand_splits.append(rand_split)
        rand_merges.append(rand_merge)
        list_xpress.append(((rand_merge+rand_split)/2 + erl_norm)/2)
        erls.append(erl/1000)
        
    with open(file_save, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['normalized erl :'] + erls)
        writer.writerow(['rand split :'] + rand_splits)
        writer.writerow(['rand merge :'] + rand_merges)
        writer.writerow(['xpress :'] + list_xpress)





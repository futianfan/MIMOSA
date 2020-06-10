import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
import timeit
from sklearn.metrics import roc_auc_score
import random
import copy 
from copy import deepcopy 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from tensorboardX import SummaryWriter
from time import time 

from loader import MoleculeDataset, mol_to_graph_data_obj_simple, graph_data_obj_to_mol_simple, allowable_features
from dataloader import DataLoaderMasking #, DataListLoader 
from model import GNN
from splitters import scaffold_split, random_split, random_scaffold_split
from util import MaskAtom
from util import similarity
from pretrain_masking import Linear 

from rdkit.Chem import AllChem, MolToSmiles
from props import drd2, penalized_logp

num_atom_type = 119 
num_edge_type = 5
similar_threshold = 0.3 
topk = 5

def masking_atom(origin_graph_data, mask_idx, mask_edge = True):
    '''
        mask_idx is integer 0~N-1.
    '''
    graph_data = copy.deepcopy(origin_graph_data) 
    graph_data.masked_atom_index = mask_idx 
    graph_data.masked_atom_label = copy.deepcopy(graph_data.x[graph_data.masked_atom_index,:]).view(1,-1)  ### shape is [1,2]
    graph_data.x[mask_idx] = torch.tensor([num_atom_type, 0])   ### modify the original node feature of the masked node 
    ##### atom #####
    ##### edge #####
    if mask_edge:
        connected_edge_indices = []
        for bond_idx, (u, v) in enumerate(graph_data.edge_index.cpu().numpy().T):
            if graph_data.masked_atom_index in set((u, v)) and bond_idx not in connected_edge_indices:
                connected_edge_indices.append(bond_idx)

        if len(connected_edge_indices) > 0:
            # create mask edge labels by copying bond features of the bonds connected to
            # the mask atoms
            mask_edge_labels_list = []
            for bond_idx in connected_edge_indices[::2]: # because the
                # edge ordering is such that two directions of a single
                # edge occur in pairs, so to get the unique undirected
                # edge indices, we take every 2nd edge index from list
                mask_edge_labels_list.append(graph_data.edge_attr[bond_idx].view(1, -1))

            graph_data.mask_edge_label = torch.cat(mask_edge_labels_list, dim=0)
            # modify the original bond features of the bonds connected to the mask atoms
            for bond_idx in connected_edge_indices:
                graph_data.edge_attr[bond_idx] = torch.tensor([num_edge_type, 0])

            graph_data.connected_edge_indices = torch.tensor(connected_edge_indices[::2])
        else:
            graph_data.mask_edge_label = torch.empty((0, 2)).to(torch.int64)
            graph_data.connected_edge_indices = torch.tensor(connected_edge_indices).to(torch.int64)
    return graph_data 


def add_atom(origin_graph_data, add_idx, mask_edge = True):
    '''
        add_idx is integer 0~N-1.
    '''
    graph_data = copy.deepcopy(origin_graph_data) 
    graph_data.add_atom_index = add_idx 
    num_atom = graph_data.x.shape[0]
    graph_data.masked_atom_index = num_atom 
    masked_atom_feature =  torch.tensor([num_atom_type, 0]).view(1,-1)  
    graph_data.x = torch.cat([graph_data.x, masked_atom_feature], 0) 
    ##### atom #####
    ##### edge #####
    if mask_edge:
        edge_connect_list = [(num_atom, add_idx), (add_idx, num_atom)]
        edge_connect_tensor = torch.tensor(np.array(edge_connect_list).T, dtype=torch.long)
        graph_data.edge_index = torch.cat([graph_data.edge_index, edge_connect_tensor], 1)

        edge_attr_tensor = torch.tensor([num_edge_type, 0]).view(1,-1)
        edge_attr_tensor = torch.cat([edge_attr_tensor, edge_attr_tensor], 0)
        graph_data.edge_attr = torch.cat([graph_data.edge_attr, edge_attr_tensor], 0)

    return graph_data 


def delete_atom_generate_smiles(origin_graph_data, delete_idx):
    '''
        delete_idx is integer 0~N-1.
    '''
    graph_data = copy.deepcopy(origin_graph_data) 
    graph_data.x = torch.cat([graph_data.x[:delete_idx,:], graph_data.x[delete_idx+1:,:]], 0)
    edge_num = graph_data.edge_index.shape[1]
    remaining_edge_idx = list(filter(lambda i:True if delete_idx not in list(graph_data.edge_index[:,i].numpy()) else False, range(edge_num)))
    graph_data.edge_index = torch.cat([graph_data.edge_index[:,i].view(-1,1) for i in remaining_edge_idx], 1)
    graph_data.edge_attr = torch.cat([graph_data.edge_attr[i,:].view(1,-1) for i in remaining_edge_idx], 0)
    f = lambda x: x-1 if x > delete_idx else x
    for i in range(graph_data.edge_index.shape[0]):
        for j in range(graph_data.edge_index.shape[1]):
            graph_data.edge_index[i][j] = f(graph_data.edge_index[i][j])
    try:
        mol = graph_data_obj_to_mol_simple(graph_data.x, graph_data.edge_index, graph_data.edge_attr)
        smiles = MolToSmiles(mol)
        return smiles 
    except:
        return None 


def atom_GNN_predict(graph_data, GNN_model, linear_pred_atoms, topk = 5):
    '''
        return a set of atom index 

        param: 
            graph_data.x, 
            graph_data.edge_index, 
            graph_data.edge_attr, 
            graph_data.masked_atom_index
    '''
    GNN_model.eval()
    linear_pred_atoms.eval()
    node_rep = GNN_model(graph_data.x, graph_data.edge_index, graph_data.edge_attr)
    pred_node = linear_pred_atoms(node_rep[graph_data.masked_atom_index])
    sorted_idx = list(pred_node.argsort().numpy())
    sorted_idx = sorted_idx[::-1]
    sorted_idx = sorted_idx[:topk]
    sorted_idx = set(sorted_idx)

    return sorted_idx 


def generate_new_smiles_for_add(origin_graph_object, graph_data, sorted_idx):
    smiles_list = []
    for node_label in sorted_idx:
        for possible_bonds in allowable_features['possible_bonds']:
            for possible_bond_dirs in allowable_features['possible_bond_dirs']:

                generated_graph = copy.deepcopy(graph_data)
                generated_graph.x[graph_data.masked_atom_index, 0] = torch.tensor(node_label) 
                generated_graph.edge_attr[-2,0] = possible_bonds 
                generated_graph.edge_attr[-2,1] = possible_bond_dirs 
                generated_graph.edge_attr[-1,0] = possible_bonds 
                generated_graph.edge_attr[-1,1] = possible_bond_dirs 
                #masked_atom_feature = torch.tensor([num_atom_type, 0]).view(1,-1)  
                #generated_graph.x = torch.cat([generated_graph.x, masked_atom_feature], 0)
                try:
                    mol = graph_data_obj_to_mol_simple(generated_graph.x, generated_graph.edge_index, generated_graph.edge_attr)
                    smiles = MolToSmiles(mol)
                    smiles_list.append(smiles)
                except:
                    continue 
    return smiles_list


def generate_new_smiles_for_mask(origin_graph_object, graph_data, sorted_idx):
    smiles_list = []
    for node_label in sorted_idx:
        generated_graph = copy.deepcopy(origin_graph_object)
        generated_graph.x[graph_data.masked_atom_index, 0] = torch.tensor(node_label) 
        try:
            mol = graph_data_obj_to_mol_simple(generated_graph.x, generated_graph.edge_index, generated_graph.edge_attr)
            smiles = MolToSmiles(mol)
            smiles_list.append(smiles)
        except:
            continue 
    return smiles_list





def arg_param():
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--mask_rate', type=float, default=0.15,
                        help='dropout ratio (default: 0.15)')
    parser.add_argument('--mask_edge', type=int, default=0,
                        help='whether to mask edges or not together with atoms')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features are combined across layers. last, sum, max or concat')
    parser.add_argument('--dataset', type=str, default = 'zinc_standard_agent', help='root directory of dataset for pretraining')
    parser.add_argument('--output_model_file', type=str, default = 'trained_model/mask_node', help='filename to output the model')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--seed', type=int, default=0, help = "Seed for splitting dataset.")
    parser.add_argument('--num_workers', type=int, default = 8, help='number of workers for dataset loading')
    args = parser.parse_args()
    return args 


def set_model_and_load_model(args, device, load_epoch = 2):
    #set up models, one for pre-training and one for context embeddings
    model = GNN(args.num_layer, args.emb_dim, JK = args.JK, drop_ratio = args.dropout_ratio, gnn_type = args.gnn_type).to(device)
    linear_pred_atoms = Linear(args.emb_dim, 119, device)
    linear_pred_bonds = Linear(args.emb_dim, 4, device)
    
    model.load_state_dict(torch.load(args.output_model_file + "_epoch" + str(load_epoch) + ".pth")) 
    linear_pred_atoms.load_state_dict(torch.load(args.output_model_file + "_epoch_" + str(load_epoch) + "_linear_atom.pth"))
    linear_pred_bonds.load_state_dict(torch.load(args.output_model_file + "_epoch_" + str(load_epoch) + "_linear_bond.pth")) 
    return model, linear_pred_atoms, linear_pred_bonds

def single_iteration(input_smiles, model_list, topk = 3, similarity_threshold = 0.3, is_print = False):
    rdkit_mol = AllChem.MolFromSmiles(input_smiles)
    data = mol_to_graph_data_obj_simple(rdkit_mol)  
    model, linear_pred_atoms = model_list  
    ###   data_x, data_edge_index, data_edge_attr = data.x, data.edge_index, data.edge_attr 
    ###   shape: [24, 2], [2, 52], [52, 2]   24--#atom; 52--#bond; 

    num_atoms = data.x.size()[0]
    mask_edge = True
    whole_smiles = []
    proposal_smiles = [] 
    if is_print:
        t1 = time()


    ##### add atom 
    for mask_idx in range(num_atoms):
        graph_data = add_atom(data, mask_idx, mask_edge)
        sorted_idx = atom_GNN_predict(graph_data, model, linear_pred_atoms, topk)
        smiles_list = generate_new_smiles_for_add(data, graph_data, sorted_idx)
        whole_smiles.extend(smiles_list)
    if is_print:
        t2 = time()
        print("\t\tsingle_iteration::1 cost {:1.2f} seconds".format(t2-t1)) 



    #### replace atom 
    for mask_idx in range(num_atoms):
        graph_data = masking_atom(data, mask_idx, mask_edge)
        sorted_idx = atom_GNN_predict(graph_data, model, linear_pred_atoms, topk)
        smiles_list = generate_new_smiles_for_mask(data, graph_data, sorted_idx)
        whole_smiles.extend(smiles_list)
    if is_print:
        t3 = time()
        print("\t\tsingle_iteration::2 cost {:1.2f} seconds".format(t3-t2)) 



    #### delete atom 
    for delete_idx in range(num_atoms):
        smiles = delete_atom_generate_smiles(data, delete_idx)
        if smiles is not None:
            whole_smiles.extend(smiles)
    if is_print:
        t4 = time()
        print("\t\tsingle_iteration::3 cost {:1.2f} seconds".format(t4-t3)) 



    whole_smiles = list(set(whole_smiles) - set([input_smiles]))
    for smiles in whole_smiles:
        try:
            sim = similarity(smiles, input_smiles)
            if (sim < similarity_threshold):
                continue 
            proposal_smiles.append((smiles, sim))
        except exception as e:
            continue
    if is_print:
        t5 = time() 
        print("\t\tsingle_iteration::4 cost {:1.2f} seconds".format(t5-t4)) 

    #print("valid rate is {:2.2f} %, valid number is {:5d}".format(len(proposal_smiles) / max(1,len(whole_smiles)) * 100, 
    #        len(proposal_smiles)))
    sim_value = [sim for smiles, sim in proposal_smiles]
    smiles_list = [smiles for smiles, sim in proposal_smiles]
    return smiles_list 


def evaluate(input_smiles, candidate_smiles, input_drd, input_logp, coefficient_similarity = 1.0, coefficient_drd = 0.3, coefficient_qed = 0.3, coefficient_logp = 0.3):
    score = similarity(input_smiles, candidate_smiles) * coefficient_similarity \
           + (drd2(candidate_smiles) - input_drd) * coefficient_drd \
           + (penalized_logp(candidate_smiles) - input_logp) * coefficient_logp 
    return score 

def judge(input_smiles, candidate_smiles, input_drd, input_logp, threshold_similarity = 0.3, threshold_qed = 0.82, drd_gap_threshold = 0.1, logp_gap_threshold = 0.3, \
             threshold_drd = 0.8):
    sim = similarity(input_smiles, candidate_smiles)
    drd_score = drd2(candidate_smiles)
    logp_score = penalized_logp(candidate_smiles)
    return sim >= threshold_similarity \
            and (drd_score >= threshold_drd or drd_score - input_drd >=drd_gap_threshold) \
            and (logp_score - input_logp > logp_gap_threshold)


def remove_repetition(current_list, existing_set):
    return list(set(current_list) - existing_set)



def multiple_iteration(input_smiles, model_list, topk = 5, max_smiles = 20, iteration = 2, is_sort=True, is_print=True):
    ###  def single_iteration(input_smiles, model_list, topk = 5)
    print("evaluate", input_smiles)
    candidate_smiles_list = [input_smiles] 
    existing_set = set()
    score_list = []
    success_set = set()
    input_drd = drd2(input_smiles)
    input_logp = penalized_logp(input_smiles)
    for i in range(iteration):
        print("iteration ", i+1)
        smiles_list = candidate_smiles_list
        candidate_smiles_list = []

        t1 = time()
        for smiles in smiles_list:
            candidate_smiles_sublist = single_iteration(smiles, model_list, topk)
            candidate_smiles_list.extend(candidate_smiles_sublist)
        candidate_smiles_list = remove_repetition(candidate_smiles_list, existing_set)
        candidate_smiles_list = list(success_set) + candidate_smiles_list
        candidate_smiles_list = list(set(candidate_smiles_list))
        #### output: candidate_smiles_list 
        if is_print:
            t2 = time()
            print("\tsingle_iteration cost {:5d} seconds".format(int(t2-t1))) 

        ### sort by posterior 
        if is_sort:
            candidate_smiles_list = [(smiles, evaluate(input_smiles, smiles, \
                                                        input_drd = input_drd, input_logp = input_logp)) \
                                      for smiles in candidate_smiles_list]
            candidate_smiles_list.sort(key=lambda x:x[1], reverse=True)

        #### truncate 
        candidate_smiles_list = candidate_smiles_list[:max_smiles]

        #### record avg score
        if is_sort:
            candidate_smiles_score_list = [score for smiles,score in candidate_smiles_list]
            score_list.append(np.mean(candidate_smiles_score_list))
            candidate_smiles_list = [smiles for smiles,score in candidate_smiles_list]
        
        existing_set.union(set(candidate_smiles_list))

        
        success_sub_list = []
        for candidate_smiles in candidate_smiles_list:
            if judge(input_smiles, candidate_smiles, input_drd = input_drd, input_logp = input_logp):
                success_sub_list.append(candidate_smiles)
        if len(success_sub_list) > 0:
            success_set = success_set.union(set(success_sub_list))
            print("\t\t success!", len(success_set))
        if is_print:
            t3 = time()
            print("\t others cost {:5d} seconds".format(int(t3-t2)))

        if len(success_set) > 3:
            print("========================================= SUCCESS =========================================")
            return success_set
    return success_set


def test_single_smiles(input_smiles, model_list):
    drd_score = drd2(input_smiles)
    logp_score = penalized_logp(input_smiles)
    success_list = multiple_iteration(input_smiles, model_list)
    for smiles in success_list:
        print(smiles, "similar: {:1.3f}; origin drd: {:1.3f}; drd improvement: {:1.2f}, logp improvement: {:2.2f}".format(\
                similarity(input_smiles, smiles), drd_score,\
                drd2(smiles) - drd_score, \
                penalized_logp(smiles) - logp_score))
    return len(success_list)


def results_append(filename, smiles):
    import os 
    if os.path.exists(filename):
        with open(filename, 'r') as fin:
            lines = fin.readlines() 
    else:
        lines = []
    smiles_set = set([line.strip() for line in lines])
    with open(filename, 'w') as fout:
        for line in lines:
            fout.write(line)
        if smiles not in smiles_set:
            fout.write(smiles + "\n")
    return 


def main():
    args = arg_param()

    ### basic setting  
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    model, linear_pred_atoms, linear_pred_bonds = set_model_and_load_model(args, device, load_epoch = 7)
    model_list = (model, linear_pred_atoms)

    with open("drd_plogp_test_1k.txt", 'r') as fin:
        lines = fin.readlines() 
    for line in lines:
        input_smiles = line.strip() 
        ll = test_single_smiles(input_smiles, model_list)
        if (ll >= 1):
            results_append("drd_plogp_success.txt", input_smiles)

if __name__ == "__main__":

    main()



'''
    smiles = "CC(=O)Nc1ccc(OC[C@H](O)CN2CCc3sccc3C2)cc1" 
    qed_score = qed(smiles)
    #logp_score = penalized_logp(smiles)
    #drd_score = drd2(smiles)
'''











































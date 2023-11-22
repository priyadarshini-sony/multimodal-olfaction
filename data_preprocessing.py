from __future__ import print_function

import os
import pickle
import random
# from collections import Counter

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import RepeatedMultilabelStratifiedKFold

from mordred import Calculator, descriptors
from rdkit import Chem
import rdkit
import networkx as nx

random.seed(2023)

def extract_mordred(smiles, save=False):
    """Given a list of smiles, extract mordred features

    Args:
        smiles (list): list of smiles
        save (bool, optional): save if true. Defaults to False.

    Returns:
        dataframe: a dataframe of features
    """
    features = []
    calc = Calculator(descriptors, ignore_3D=True)
    print(len(calc.descriptors))
    mols = [Chem.MolFromSmiles(smi) for smi in smiles]
    features = calc.pandas(mols)
    features["smiles"] = smiles
    print(features)
    if save:
        features.to_csv("mordred_features.csv")
    return features


def label_dist(y):
    """compute label distribution of multilabel dataset

    Args:
        y (array): multilabels of all samples

    Returns:
        array: distribution of each label
    """
    lst = []
    for i in range(y.shape[1]):
        cur_col = np.mean(y[:, i])
        lst.append(cur_col)
    lst = np.array(lst)
    return lst.T


def split_train_test(splits, repeats, X, y, save_dir):
    """stratified split of multilabel dataset into train and test

    Args:
        n_splits (int): n-fold
        n_repeats (int): number of times to repeat
        X (array): features
        y (array): labels
    """
    rmskf = RepeatedMultilabelStratifiedKFold(n_splits=splits, n_repeats=repeats)
    for i, (train_index, test_index) in enumerate(rmskf.split(X, y)):

        sv_path = f"{save_dir}/split_{i}.pkl"

        if os.path.exists(sv_path):
            continue

        print("TRAIN:", train_index, "TEST:", test_index)
        print("Len TRAIN:", len(train_index), "TEST:", len(test_index))

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # check label distribution
        print(label_dist((y_train)))
        print(label_dist((y_test)))

        data = {}
        data["X_train"] = X_train
        data["X_test"] = X_test
        data["y_train"] = y_train
        data["y_test"] = y_test
        data["train_index"] = train_index
        data["test_index"] = test_index

        with open(sv_path, "wb") as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def save_train_test(sv_dir):
    """save train and test data"""

    labels = pd.read_csv(f"{sv_dir}/labels.csv")
    feat = pd.read_csv(f"{sv_dir}/mordred_prune.csv")
    # print(feat, "\n", labels)

    del feat["smiles"]
    del labels["Unnamed: 0"]
    del labels["cid"]
    del labels["smiles"]

    X = feat.to_numpy()
    y = labels.to_numpy()

    split_train_test(5, 3, X, y, sv_dir)
    print("Train and test data creation completed")

def save_Graph_train_test(sv_dir):
    """save train and test data for graph"""

    labels = pd.read_csv(f"{sv_dir}/labels.csv")
    smiles = labels[['cid','smiles']]
    dict_CID_smile = dict(zip(smiles['cid'],smiles['smiles']))

    y = labels.iloc[:,3:].to_numpy()

    data = Qm9(smiles['cid'].values, edge_transform=qm9_edges, e_representation="raw_distance")
    feat = []
    for i in range(len(data)):
        feat.append(data[i][0])
    X= np.asarray(data,dtype=object)
    
    split_train_test(5, 3, X, y, sv_dir)
    print("Train and test graph data creation completed")

# save_train_test("data")

# def load_data(data_dir, max_run):
#     X_train, X_test, y_train, y_test = [], [], [], []
#     for n_run in range(max_run):
#         # file_name = data_dir + "split_" + str(n_run) + ".pkl"
#         file_name = data_dir + "split_" + str(n_run) + "_smiles.pkl"
#         with open(file_name, 'rb') as handle:  
#             b = pickle.loads(handle.read())
#         #with open(file_name, "rb", errors='ignore') as handle:
#         #    b = pickle.load(handle)
#         # print('-'*50)
#         # print('checking', b["X_train"].shape)
#         # print('-'*50)
#         X_train.append(b["X_train"])
#         X_test.append(b["X_test"])
#         y_train.append(b["y_train"])
#         y_test.append(b["y_test"])
#         print("Data loading completed")

#     return X_train, X_test, y_train, y_test

def load_data(data_dir, max_run, percent):
    X_train, X_test, y_train, y_test = [], [], [], []
    for n_run in range(max_run):
        file_name = data_dir + "split_" + str(n_run) + f"_smiles_train{percent}.pkl"
        with open(file_name, 'rb') as handle:  
            b = pickle.loads(handle.read())
        #with open(file_name, "rb", errors='ignore') as handle:
        #    b = pickle.load(handle)
        X_train.append(b["X_train"])
        X_test.append(b["X_test"])
        y_train.append(b["y_train"])
        y_test.append(b["y_test"])
        print("Data loading completed")

    return X_train, X_test, y_train, y_test

def load_data_with_val(data_dir, max_run, percent):
    X_train, X_test, y_train, y_test = [], [], [], []
    X_val, y_val = [], []
    for n_run in range(max_run):
        file_name = data_dir + "split_" + str(n_run) + f"_smiles_train{percent}_val.pkl"
        with open(file_name, 'rb') as handle:  
            b = pickle.loads(handle.read())
        #with open(file_name, "rb", errors='ignore') as handle:
        #    b = pickle.load(handle)
        X_train.append(b["X_train"])
        X_val.append(b["X_val"])
        X_test.append(b["X_test"])
        y_train.append(b["y_train"])
        y_val.append(b["y_val"])
        y_test.append(b["y_test"])
        print("Data loading completed")

    return X_train, X_val, X_test, y_train, y_val, y_test


def qm9_edges(g):
    """load MPNN dataset

    Args:
        g: graphs of molecules
    """
        
    remove_edges = []
    e={}    
    for n1, n2, d in g.edges(data=True):
        e_t = []
        # Raw distance function
        if d['b_type'] is None:
            remove_edges += [(n1, n2)]
        else:
            #e_t.append(d['distance'])
            e_t += [int(d['b_type'] == x) for x in [rdkit.Chem.rdchem.BondType.SINGLE, rdkit.Chem.rdchem.BondType.DOUBLE,
                                                    rdkit.Chem.rdchem.BondType.TRIPLE, rdkit.Chem.rdchem.BondType.AROMATIC]]
        if e_t:
            e[(n1, n2)] = e_t
    for edg in remove_edges:
        g.remove_edge(*edg)    
    
    return nx.to_numpy_matrix(g), e
    
def qm9_nodes(g, hydrogen=False):
    h = []
    for n, d in g.nodes(data=True): 
        h_t = []
        # Atom type (One-hot H, C, N, O F)
        h_t += [int(d['a_type'] == x) for x in ['H', 'C', 'N', 'O', 'F']]
        # Atomic number
        h_t.append(d['a_num'])
        # Acceptor
        h_t.append(d['acceptor'])
        # Donor
        h_t.append(d['donor'])
        # Aromatic
        h_t.append(int(d['aromatic']))
        # If number hydrogen is used as a
        if hydrogen:
            h_t.append(d['num_h'])
        h.append(h_t)
    return h

def xyz_graph_reader(CID):
    smiles = dict_CID_smile[CID]
    m = Chem.MolFromSmiles(smiles)
    m = Chem.AddHs(m)
   
    g = nx.Graph()
    # Create nodes
    for i in range(0, m.GetNumAtoms()):
        atom_i = m.GetAtomWithIdx(i)

        g.add_node(i, a_type=atom_i.GetSymbol(), a_num=atom_i.GetAtomicNum(), acceptor=0, donor=0,
                   aromatic=atom_i.GetIsAromatic(), hybridization=atom_i.GetHybridization(),
                   num_h=atom_i.GetTotalNumHs())


    # Read Edges
    for i in range(0, m.GetNumAtoms()):
        for j in range(0, m.GetNumAtoms()):
            e_ij = m.GetBondBetweenAtoms(i, j)
            if e_ij is not None:
                g.add_edge(i, j, b_type=e_ij.GetBondType())
            else:
                # Unbonded
                g.add_edge(i, j, b_type=None)
    
    l = labels[labels['cid'] ==CID]
    l = l.iloc[:,3:].values       
                
    return g , l

def qm9_edges(g):
    remove_edges = []
    e={}    
    for n1, n2, d in g.edges(data=True):
        e_t = []
        # Raw distance function
        if d['b_type'] is None:
            remove_edges += [(n1, n2)]
        else:
            #e_t.append(d['distance'])
            e_t += [int(d['b_type'] == x) for x in [rdkit.Chem.rdchem.BondType.SINGLE, rdkit.Chem.rdchem.BondType.DOUBLE,
                                                    rdkit.Chem.rdchem.BondType.TRIPLE, rdkit.Chem.rdchem.BondType.AROMATIC]]
        if e_t:
            e[(n1, n2)] = e_t
    for edg in remove_edges:
        g.remove_edge(*edg)    
    
    return nx.to_numpy_matrix(g), e
    
def qm9_nodes(g, hydrogen=False):
    h = []
    for n, d in g.nodes(data=True): 
        h_t = []
        # Atom type (One-hot H, C, N, O F)
        h_t += [int(d['a_type'] == x) for x in ['H', 'C', 'N', 'O', 'F']]
        # Atomic number
        h_t.append(d['a_num'])
        # Acceptor
        h_t.append(d['acceptor'])
        # Donor
        h_t.append(d['donor'])
        # Aromatic
        h_t.append(int(d['aromatic']))
        # If number hydrogen is used as a
        if hydrogen:
            h_t.append(d['num_h'])
        h.append(h_t)
    return h

def xyz_graph_reader(CID):
    smiles = dict_CID_smile[CID]
    m = Chem.MolFromSmiles(smiles)
    m = Chem.AddHs(m)
   
    g = nx.Graph()
    # Create nodes
    for i in range(0, m.GetNumAtoms()):
        atom_i = m.GetAtomWithIdx(i)

        g.add_node(i, a_type=atom_i.GetSymbol(), a_num=atom_i.GetAtomicNum(), acceptor=0, donor=0,
                   aromatic=atom_i.GetIsAromatic(), hybridization=atom_i.GetHybridization(),
                   num_h=atom_i.GetTotalNumHs())


    # Read Edges
    for i in range(0, m.GetNumAtoms()):
        for j in range(0, m.GetNumAtoms()):
            e_ij = m.GetBondBetweenAtoms(i, j)
            if e_ij is not None:
                g.add_edge(i, j, b_type=e_ij.GetBondType())
            else:
                # Unbonded
                g.add_edge(i, j, b_type=None)
    
    l = labels[labels['cid'] ==CID]
    l = l.iloc[:,3:].values       
                
    return g , l

class Qm9():
    # Constructor
    def __init__(self, idx, vertex_transform=qm9_nodes, edge_transform=qm9_edges,
                 target_transform=None, e_representation='raw_distance'):
        self.idx = idx
        self.vertex_transform = vertex_transform
        self.edge_transform = edge_transform
        self.target_transform = target_transform
        self.e_representation = e_representation

    def __getitem__(self,index):
        
        g, target = xyz_graph_reader(self.idx[index])
        if self.vertex_transform is not None:
            h = self.vertex_transform(g)

        if self.edge_transform is not None:
            g, e = self.edge_transform(g)

#         if self.target_transform is not None:
#             target = self.target_transform(target)

        #g：adjacent matrix
        #h：node properties（list of list）
        #e：diction，key:edge，value:properties
        return (g, h, e)
        #return (g, h, e), target

    def __len__(self):
        return len(self.idx)

#     def set_target_transform(self, target_transform):
#         self.target_transform = target_transform

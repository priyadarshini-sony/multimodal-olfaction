from __future__ import print_function

import random
import warnings
from collections import defaultdict

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
#from tensorflow.keras import backend as K
from torch.utils.data import Dataset
from keras.callbacks import Callback
# from keras.layers import Dense
# from keras.models import Sequential

from sklearn.decomposition import PCA
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

import os
import csv
import math
import time
import random
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler

from torch_scatter import scatter
from torch_geometric.data import Data, Dataset#, DataLoader
from torch.utils.data import DataLoader

from torch_geometric.data import Data, Batch

import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from rdkit import RDLogger                                                                                                                                                               
RDLogger.DisableLog('rdApp.*')  

# import os
# import argparse
# import time

random.seed(2023)

# gpu settings
use_cuda = torch.cuda.is_available()
print("gpu status ===", use_cuda)

ATOM_LIST = list(range(1,119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
BOND_LIST = [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT
]



class OlfactortyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.smiles_data = self.X[:, -1]
        self.y = y
        self.labels = y

    def __getitem__(self, index):
        mol = Chem.MolFromSmiles(self.smiles_data[index])
        mol = Chem.AddHs(mol)

        N = mol.GetNumAtoms()
        M = mol.GetNumBonds()

        type_idx = []
        chirality_idx = []
        atomic_number = []
        for atom in mol.GetAtoms():
            type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
            chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
            atomic_number.append(atom.GetAtomicNum())

        x1 = torch.tensor(type_idx, dtype=torch.long).view(-1,1)
        x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1,1)
        x = torch.cat([x1, x2], dim=-1)

        row, col, edge_feat = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])

        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)

        y = torch.tensor(self.labels[index], dtype=torch.long).view(1,-1)

        data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)

        feat = self.X[index]
        label = self.y[index]
        return feat, label, data

    def __len__(self):
        return len(self.y)


def class_weights(labels):
    """compute class weights

    Args:
        labels (array): multilabel array

    Returns:
        dict: positive and negative weights for each class
    """
    class_weights = {}
    positive_weights = {}
    negative_weights = {}

    for k in range(labels.shape[1]):
        positives = sum(labels[:, k] == 1)
        negatives = labels.shape[0] - positives
        # print(
        #     "{}:\tPositive Samples: {}\t\tNegative Samples: {}".format(
        #         k, positives, negatives
        #     )
        # )
        positive_weights[k] = labels.shape[0] / (2 * positives)
        negative_weights[k] = labels.shape[0] / (2 * negatives)

    class_weights["positive_weights"] = positive_weights
    class_weights["negative_weights"] = negative_weights
    # print("\class weight: {}".format(class_weights))
    return class_weights


def custom_loss(class_weights, cfg):
    def loss(y_true, y_logit):
        """
        Multi-label cross-entropy
        * Required "Wp", "Wn" as positive & negative class-weights
        y_true: true value
        y_logit: predicted value
        """
        Wp = np.array(list(class_weights["positive_weights"].values())).astype(float)
        # print(Wp, type(Wp))
        Wp = Wp * cfg.train.pos_weight

        Wn = np.array(list(class_weights["negative_weights"].values())).astype(float)
        Wn = Wn * cfg.train.neg_weight

        Wpl = torch.Tensor(Wp).to("cuda")
        Wnl = torch.Tensor(Wn).to("cuda")

        first_term = Wpl * y_true * torch.log(y_logit + 1e-10)

        second_term = Wnl * (1 - y_true) * torch.log(1 - y_logit + 1e-10)
        loss_value = torch.mean(first_term + second_term)

        return loss_value

    return loss

def get_eval_report_embed(y_true, y_pred):
    warnings.filterwarnings("ignore")

    """Evaluation class-wise precision, recall, and F1-score.

    Args:
        y_test (np.array): true label
        y_pred (np.array): predicted label

    Returns:
        result 2D dictionary with class-wise performance metrics.
    """
    # keys = ["precision", "recall", "f1-score"]
    y_score = y_pred
    y_pred = (y_pred >= 0.5).astype(np.float32)

    # label_name = pd.read_csv("./data_numeric/labels.csv")
    label_name = pd.read_csv("./data_multi/labels.csv")
    label_name = np.array(list(label_name.columns))

    acc_report = {}
    weight, f1 = [], []
    precision, recall, auroc = [], [], []
    count = 3

    for i in range(y_pred.shape[1]):

        p, r, f, s = precision_recall_fscore_support(
            y_true[:, i], y_pred[:, i], average=None
        )
        roc = roc_auc_score(y_true[:, i], y_score[:, i], average=None)

        weight.append(s[1])
        f1.append(f[1])
        precision.append(p[1])
        recall.append(r[1])
        auroc.append(roc)
        acc_report['embed-' + label_name[count]] = p[1], r[1], f[1], s[1], roc
        count += 1

    weight = [k / (y_true.shape[0]) for k in weight]
    weight = np.array(weight)
    f1 = np.array(f1)
    acc_report["embed-f1-macro"] = np.average(f1)
    acc_report["embed-f1-micro"] = np.average(f1, weights=weight)
    acc_report["embed-precision-macro"] = np.average(precision)
    acc_report["embed-recall-macro"] = np.average(recall)
    acc_report["embed-auroc-macro"] = np.average(auroc)


    acc_report["embed-var-f1-macro"] = np.var(f1)
    # acc_report["var-f1-micro"] = np.var(f1, weights=weight)
    acc_report["embed-var-precision-macro"] = np.var(precision)
    acc_report["embed-var-recall-macro"] = np.var(recall)
    acc_report["embed-var-auroc-macro"] = np.var(auroc)

    acc_report["embed-var-auroc-macro"] = np.var(auroc)

    # print(acc_report)
    return acc_report

def get_eval_report_molclr(y_true, y_pred):
    warnings.filterwarnings("ignore")

    """Evaluation class-wise precision, recall, and F1-score.

    Args:
        y_test (np.array): true label
        y_pred (np.array): predicted label

    Returns:
        result 2D dictionary with class-wise performance metrics.
    """
    # keys = ["precision", "recall", "f1-score"]
    y_score = y_pred
    y_pred = (y_pred >= 0.5).astype(np.float32)

    # label_name = pd.read_csv("./data_numeric/labels.csv")
    label_name = pd.read_csv("./data_multi/labels.csv")
    label_name = np.array(list(label_name.columns))

    acc_report = {}
    weight, f1 = [], []
    precision, recall, auroc = [], [], []
    count = 3

    for i in range(y_pred.shape[1]):

        p, r, f, s = precision_recall_fscore_support(
            y_true[:, i], y_pred[:, i], average=None
        )
        roc = roc_auc_score(y_true[:, i], y_score[:, i], average=None)

        weight.append(s[1])
        f1.append(f[1])
        precision.append(p[1])
        recall.append(r[1])
        auroc.append(roc)
        acc_report['molclr-' + label_name[count]] = p[1], r[1], f[1], s[1], roc
        count += 1

    weight = [k / (y_true.shape[0]) for k in weight]
    weight = np.array(weight)
    f1 = np.array(f1)
    acc_report["molclr-f1-macro"] = np.average(f1)
    acc_report["molclr-f1-micro"] = np.average(f1, weights=weight)
    acc_report["molclr-precision-macro"] = np.average(precision)
    acc_report["molclr-recall-macro"] = np.average(recall)
    acc_report["molclr-auroc-macro"] = np.average(auroc)


    acc_report["molclr-var-f1-macro"] = np.var(f1)
    # acc_report["var-f1-micro"] = np.var(f1, weights=weight)
    acc_report["molclr-var-precision-macro"] = np.var(precision)
    acc_report["molclr-var-recall-macro"] = np.var(recall)
    acc_report["molclr-var-auroc-macro"] = np.var(auroc)
    acc_report["molclr-var-auroc-macro"] = np.var(auroc)

    # print(acc_report)
    return acc_report

def get_eval_report(y_true, y_pred):
    warnings.filterwarnings("ignore")

    """Evaluation class-wise precision, recall, and F1-score.

    Args:
        y_test (np.array): true label
        y_pred (np.array): predicted label

    Returns:
        result 2D dictionary with class-wise performance metrics.
    """
    # keys = ["precision", "recall", "f1-score"]
    y_score = y_pred
    y_pred = (y_pred >= 0.5).astype(np.float32)

    # label_name = pd.read_csv("./data_numeric/labels.csv")
    label_name = pd.read_csv("./data_multi/labels.csv")

    label_name = np.array(list(label_name.columns))

    acc_report = {}
    weight, f1 = [], []
    precision, recall, auroc = [], [], []
    count = 3
    # count = 0

    for i in range(y_pred.shape[1]):

        p, r, f, s = precision_recall_fscore_support(
            y_true[:, i], y_pred[:, i], average=None
        )
        roc = roc_auc_score(y_true[:, i], y_score[:, i], average=None)

        weight.append(s[1])
        f1.append(f[1])
        precision.append(p[1])
        recall.append(r[1])
        auroc.append(roc)
        acc_report[label_name[count]] = p[1], r[1], f[1], s[1], roc
        count += 1

    weight = [k / (y_true.shape[0]) for k in weight]
    weight = np.array(weight)
    f1 = np.array(f1)
    acc_report["f1-macro"] = np.average(f1)
    acc_report["f1-micro"] = np.average(f1, weights=weight)
    acc_report["precision-macro"] = np.average(precision)
    acc_report["recall-macro"] = np.average(recall)
    acc_report["auroc-macro"] = np.average(auroc)

    acc_report["auroc-macro"] = np.average(auroc)

    acc_report["var-f1-macro"] = np.var(f1)
    # acc_report["var-f1-micro"] = np.var(f1, weights=weight)
    acc_report["var-precision-macro"] = np.var(precision)
    acc_report["var-recall-macro"] = np.var(recall)
    acc_report["var-auroc-macro"] = np.var(auroc)

    acc_report["var-auroc-macro"] = np.var(auroc)

    # print(acc_report)
    return acc_report

def get_eval_report_two(y_true, y_pred1, y_pred2):
    warnings.filterwarnings("ignore")

    """Evaluation class-wise precision, recall, and F1-score.

    Args:
        y_test (np.array): true label
        y_pred (np.array): predicted label

    Returns:
        result 2D dictionary with class-wise performance metrics.
    """
    # keys = ["precision", "recall", "f1-score"]
    y_score1 = y_pred1
    y_score2 = y_pred2

    y_pred1 = (y_pred1 >= 0.5).astype(np.float32)
    y_pred2 = (y_pred2 >= 0.5).astype(np.float32)

    # label_name = pd.read_csv("./data_numeric/labels.csv")
    label_name = pd.read_csv("./data_multi/labels.csv")
    label_name = np.array(list(label_name.columns))

    acc_report = {}
    weight, f1 = [], []
    precision, recall, auroc = [], [], []
    count = 3

    auroc_idx = []

    for i in range(y_pred1.shape[1]):    

        try:
            roc1 = roc_auc_score(y_true[:, i], y_score1[:, i], average=None)
            roc2 = roc_auc_score(y_true[:, i], y_score2[:, i], average=None)
        except ValueError:
            roc1 = 0.0
            roc2 = 0.0
            print('value error')

        roc = max(roc1, roc2)
        roc_idx = 0
        if roc2 > roc1:
            roc_idx = 1
        auroc_idx.append(roc_idx)

        auroc.append(roc)
        count += 1

    acc_report["auroc-macro"] = np.average(auroc)

    return acc_report, auroc_idx


def log_eval_report(prefix, acc_report, mlflow, epoch, d1, r):
    # d1 = {}
    if epoch not in d1:
        d1[epoch] = defaultdict(dict)
    # if r not in d1[epoch]:
    #     d1[epoch][r]={}
    for k, v in acc_report.items():
        if k in [
            "loss",
            "f1-macro",
            "f1-micro",
            "precision-macro",
            "recall-macro",
            "auroc-macro",

            "var-f1-macro",
            "var-precision-macro",
            "var-recall-macro",
            "var-auroc-macro",
        ]:
            d1[epoch][f"{prefix}-{k}"][r] = v
            # mlflow.log_metric(f"{prefix}-{k}", v, step=epoch)
        else:
            v1, v2, v3, v4, v5 = v
            d1[epoch][f"{prefix}-{k}-precision"][r] = v1
            d1[epoch][f"{prefix}-{k}-recall"][r] = v2
            d1[epoch][f"{prefix}-{k}-f1"][r] = v3
            d1[epoch][f"{prefix}-{k}-support"][r] = v4
            d1[epoch][f"{prefix}-{k}-auroc"][r] = v5
    return d1


def pca(feat, factor):

    sc = StandardScaler()
    sc.fit(feat)
    X_train = sc.transform(feat)
    my_model = PCA(n_components=factor)
    X_train_pca = my_model.fit_transform(X_train)
    # print(X_train_pca.shape)
    return X_train_pca


class CustomCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print("\nL1:\t", logs, epoch)

        mlflow.log_metrics(
            {
                "train_loss": logs["loss"],
                "train_accuracy": logs["binary_accuracy"],
            },
            epoch,
        )
        
        

#### utils for MPNN ####

def collate_mpnn(batch):
    batch_sizes = np.max(np.array([[len(input_b[1]), len(input_b[1][0]), len(input_b[2]),
                                len(list(input_b[2].values())[0])]
                                if input_b[2] else
                                [len(input_b[1]), len(input_b[1][0]), 0,0]
                                for (input_b, target_b) in batch]), axis=0)    
    
    g = np.zeros((len(batch), batch_sizes[0], batch_sizes[0]))
    h = np.zeros((len(batch), batch_sizes[0], batch_sizes[1]))
    e = np.zeros((len(batch), batch_sizes[0], batch_sizes[0], batch_sizes[3]))

    target = np.zeros((len(batch), len(batch[0][1])))

    for i in range(len(batch)):

        num_nodes = len(batch[i][0][1])

        # Adjacency matrix
        g[i, 0:num_nodes, 0:num_nodes] = batch[i][0][0]

        # Node features
        h[i, 0:num_nodes, :] = batch[i][0][1]

        # Edges
        for edge in batch[i][0][2].keys():
            e[i, edge[0], edge[1], :] = batch[i][0][2][edge]
            e[i, edge[1], edge[0], :] = batch[i][0][2][edge]

        # Target
        target[i, :] = batch[i][1]

    g = torch.FloatTensor(g)
    h = torch.FloatTensor(h)
    e = torch.FloatTensor(e)
    target = torch.FloatTensor(target)

    return g, h, e, target


def collate_g(batch, follow_batch=[]):

    batch_sizes = np.max(np.array([[len(input_b[1]), len(input_b[1][0]), len(input_b[2]),
                                len(list(input_b[2].values())[0])]
                                if input_b[2] else
                                [len(input_b[1]), len(input_b[1][0]), 0,0]
                                for (input_b, target_b, _) in batch]), axis=0)    
    
    
    g = np.zeros((len(batch), batch_sizes[0], batch_sizes[0]))
    h = np.zeros((len(batch), batch_sizes[0], batch_sizes[1]))
    e = np.zeros((len(batch), batch_sizes[0], batch_sizes[0], batch_sizes[3]))
    numeric = np.zeros((len(batch), 1388))
    sequence = np.zeros((len(batch), 162, 5))

    embedding = np.zeros((len(batch), 512))

    target = np.zeros((len(batch), len(batch[0][1])))
    

    for i in range(len(batch)):

        num_nodes = len(batch[i][0][1])

        # Adjacency matrix
        g[i, 0:num_nodes, 0:num_nodes] = batch[i][0][0]

        # Node features
        h[i, 0:num_nodes, :] = batch[i][0][1]

        # Edges
        for edge in batch[i][0][2].keys():
            e[i, edge[0], edge[1], :] = batch[i][0][2][edge]
            e[i, edge[1], edge[0], :] = batch[i][0][2][edge]
        
        numeric[i, :] = batch[i][0][3]
        sequence[i, :] = batch[i][0][4]
        # Target
        target[i, :] = batch[i][1]

        embedding[i, :] = batch[i][0][5]

    g = torch.FloatTensor(g)
    h = torch.FloatTensor(h)
    e = torch.FloatTensor(e)
    numeric = torch.FloatTensor(numeric)
    sequence = torch.FloatTensor(sequence)
    target = torch.FloatTensor(target)

    embedding = torch.FloatTensor(embedding)
    datalist = [data_point[2] for data_point in batch]

    batch_data = Batch.from_data_list(datalist, follow_batch)

    return g, h, e, numeric, sequence, embedding, batch_data, target


"""
    MessageFunction.py: Propagates a message depending on two nodes and their common edge.
"""
class NNet(nn.Module):

    def __init__(self, n_in, n_out, hlayers=(128, 256, 128)):
        super(NNet, self).__init__()
        self.n_hlayers = len(hlayers)
        self.fcs = nn.ModuleList([nn.Linear(n_in, hlayers[i]) if i == 0 else
                                  nn.Linear(hlayers[i-1], n_out) if i == self.n_hlayers else
                                  nn.Linear(hlayers[i-1], hlayers[i]) for i in range(self.n_hlayers+1)])

    def forward(self, x):
        x = x.contiguous().view(-1, self.num_flat_features(x))
        for i in range(self.n_hlayers):
            x = F.relu(self.fcs[i](x))
        x = self.fcs[-1](x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


"""
    UpdateFunction.py: Updates the nodes using the previous state and the message.
"""
class MessageFunction(nn.Module):

    # Constructor
    def __init__(self, message_def='mpnn', args={}):
        super(MessageFunction, self).__init__()
        self.m_definition = ''
        self.m_function = None
        self.args = {}
        self.__set_message(message_def, args)

    # Message from h_v to h_w through e_vw
    def forward(self, h_v, h_w, e_vw, args=None):
        return self.m_function(h_v, h_w, e_vw, args)

    # Set a message function
    def __set_message(self, message_def, args={}):
        self.m_definition = message_def.lower()

        self.m_function = {
                    'duvenaud':         self.m_duvenaud,
                    'intnet':             self.m_intnet,
                    'mpnn':             self.m_mpnn,
                }.get(self.m_definition, None)

        if self.m_function is None:
            print('WARNING!: Message Function has not been set correctly\n\tIncorrect definition ' + message_def)
            quit()

        init_parameters = {
            'duvenaud': self.init_duvenaud,            
            'intnet':     self.init_intnet,
            'mpnn':     self.init_mpnn
        }.get(self.m_definition, lambda x: (nn.ParameterList([]), nn.ModuleList([]), {}))

        self.learn_args, self.learn_modules, self.args = init_parameters(args)

        self.m_size = {
                'duvenaud':     self.out_duvenaud,            
                'intnet':         self.out_intnet,
                'mpnn':         self.out_mpnn
            }.get(self.m_definition, None)

    # Get the name of the used message function
    def get_definition(self):
        return self.m_definition

    # Get the message function arguments
    def get_args(self):
        return self.args

    # Get Output size
    def get_out_size(self, size_h, size_e, args=None):
        return self.m_size(size_h, size_e, args)
    
    
    # Duvenaud et al. (2015), Convolutional Networks for Learning Molecular Fingerprints
    def m_duvenaud(self, h_v, h_w, e_vw, args):
        m = torch.cat([h_w, e_vw], 2)
        return m

    def out_duvenaud(self, size_h, size_e, args):
        return size_h + size_e

    def init_duvenaud(self, params):
        learn_args = []
        learn_modules = []
        args = {}
        return nn.ParameterList(learn_args), nn.ModuleList(learn_modules), args 

    # Battaglia et al. (2016), Interaction Networks
    def m_intnet(self, h_v, h_w, e_vw, args):
        m = torch.cat([h_v[:, None, :].expand_as(h_w), h_w, e_vw], 2)
        b_size = m.size()

        m = m.view(-1, b_size[2])

        m = self.learn_modules[0](m)
        m = m.view(b_size[0], b_size[1], -1)
        return m

    def out_intnet(self, size_h, size_e, args):
        return self.args['out']

    def init_intnet(self, params):
        learn_args = []
        learn_modules = []
        args = {}
        args['in'] = params['in']
        args['out'] = params['out']
        learn_modules.append(NNet(n_in=params['in'], n_out=params['out']))
        return nn.ParameterList(learn_args), nn.ModuleList(learn_modules), args

    # Gilmer et al. (2017), Neural Message Passing for Quantum Chemistry
    def m_mpnn(self, h_v, h_w, e_vw, opt={}):
        # Matrices for each edge
        edge_output = self.learn_modules[0](e_vw)
        edge_output = edge_output.view(-1, self.args['out'], self.args['in'])

        #h_w_rows = h_w[..., None].expand(h_w.size(0), h_v.size(1), h_w.size(1)).contiguous()
        h_w_rows = h_w[..., None].expand(h_w.size(0), h_w.size(1), h_v.size(1)).contiguous()

        h_w_rows = h_w_rows.view(-1, self.args['in'])

        h_multiply = torch.bmm(edge_output, torch.unsqueeze(h_w_rows,2))

        m_new = torch.squeeze(h_multiply)

        return m_new

    def out_mpnn(self, size_h, size_e, args):
        return self.args['out']

    def init_mpnn(self, params):
        learn_args = []
        learn_modules = []
        args = {}

        args['in'] = params['in']
        args['out'] = params['out']

        # Define a parameter matrix A for each edge label.
        learn_modules.append(NNet(n_in=params['edge_feat'], n_out=(params['in']*params['out'])))

        return nn.ParameterList(learn_args), nn.ModuleList(learn_modules), args

class UpdateFunction(nn.Module):
    # Constructor
    def __init__(self, update_def='nn', args={}):
        super(UpdateFunction, self).__init__()
        self.u_definition = ''
        self.u_function = None
        self.args = {}
        self.__set_update(update_def, args)

    # Update node hv given message mv
    def forward(self, h_v, m_v, opt={}):
        return self.u_function(h_v, m_v, opt)

    # Set update function
    def __set_update(self, update_def, args):
        self.u_definition = update_def.lower()

        self.u_function = {
                    'duvenaud':         self.u_duvenaud,            
                    'intnet':             self.u_intnet,
                    'mpnn':             self.u_mpnn
                }.get(self.u_definition, None)

        if self.u_function is None:
            print('WARNING!: Update Function has not been set correctly\n\tIncorrect definition ' + update_def)

        init_parameters = {
            'duvenaud':         self.init_duvenaud,            
            'intnet':             self.init_intnet,
            'mpnn':             self.init_mpnn
        }.get(self.u_definition, lambda x: (nn.ParameterList([]), nn.ModuleList([]), {}))

        self.learn_args, self.learn_modules, self.args = init_parameters(args)

    # Get the name of the used update function
    def get_definition(self):
        return self.u_definition

    # Get the update function arguments
    def get_args(self):
        return self.args
    
    # Duvenaud
    def u_duvenaud(self, h_v, m_v, opt):

        param_sz = self.learn_args[0][opt['deg']].size()
        parameter_mat = torch.t(self.learn_args[0][opt['deg']])[None, ...].expand(m_v.size(0), param_sz[1], param_sz[0])
        
        #print(parameter_mat.size())
        #print(m_v.size())
        #print(torch.transpose(m_v.unsqueeze(-2), 1, 2).size())

        #aux = torch.bmm(parameter_mat, torch.transpose(m_v, 1, 2))
        aux = torch.bmm(parameter_mat, torch.transpose(m_v.unsqueeze(-2), 1, 2))

        return torch.transpose(torch.nn.Sigmoid()(aux), 1, 2)

    def init_duvenaud(self, params):
        learn_args = []
        learn_modules = []
        args = {}

        # Filter degree 0 (the message will be 0 and therefore there is no update
        args['deg'] = [i for i in params['deg'] if i!=0]
        args['in'] = params['in']
        args['out'] = params['out']

        # Define a parameter matrix H for each degree.
        learn_args.append(torch.nn.Parameter(torch.randn(len(args['deg']), args['in'], args['out'])))

        return nn.ParameterList(learn_args), nn.ModuleList(learn_modules), args    

    # Battaglia et al. (2016), Interaction Networks
    def u_intnet(self, h_v, m_v, opt):
        if opt['x_v'].ndimension():
            input_tensor = torch.cat([h_v, opt['x_v'], torch.squeeze(m_v)], 1)
        else:
            input_tensor = torch.cat([h_v, torch.squeeze(m_v)], 1)

        return self.learn_modules[0](input_tensor)

    def init_intnet(self, params):
        learn_args = []
        learn_modules = []
        args = {}

        args['in'] = params['in']
        args['out'] = params['out']

        learn_modules.append(NNet(n_in=params['in'], n_out=params['out']))

        return nn.ParameterList(learn_args), nn.ModuleList(learn_modules), args

    def u_mpnn(self, h_v, m_v, opt={}):
        h_in = h_v.view(-1,h_v.size(2))
        if len(m_v.size()) <3:
            m_v=torch.unsqueeze(m_v,0)
        m_in = m_v.view(-1,m_v.size(2))
        h_new = self.learn_modules[0](m_in[None,...],h_in[None,...])[0] # 0 or 1???
        return torch.squeeze(h_new).view(h_v.size())

    def init_mpnn(self, params):
        learn_args = []
        learn_modules = []
        args = {}

        args['in_m'] = params['in_m']
        args['out'] = params['out']

        # GRU
        learn_modules.append(nn.GRU(params['in_m'], params['out']))

        return nn.ParameterList(learn_args), nn.ModuleList(learn_modules), args


"""
    MessageFunction.py: Propagates a message depending on two nodes and their common edge.
"""


#dtype = torch.cuda.FloatTensor
dtype = torch.FloatTensor


class ReadoutFunction(nn.Module):

    # Constructor
    def __init__(self, readout_def='nn', args={}):
        super(ReadoutFunction, self).__init__()
        self.r_definition = ''
        self.r_function = None
        self.args = {}
        self.__set_readout(readout_def, args)

    # Readout graph given node values at las layer
    def forward(self, h_v):
        return self.r_function(h_v)

    # Set a readout function
    def __set_readout(self, readout_def, args):
        self.r_definition = readout_def.lower()

        self.r_function = {
                    'duvenaud': self.r_duvenaud,            
                    'intnet':     self.r_intnet,
                    'mpnn':     self.r_mpnn
                }.get(self.r_definition, None)

        if self.r_function is None:
            print('WARNING!: Readout Function has not been set correctly\n\tIncorrect definition ' + readout_def)
            quit()

        init_parameters = {
            'duvenaud': self.init_duvenaud,            
            'intnet':     self.init_intnet,
            'mpnn':     self.init_mpnn
        }.get(self.r_definition, lambda x: (nn.ParameterList([]), nn.ModuleList([]), {}))

        self.learn_args, self.learn_modules, self.args = init_parameters(args)

    # Get the name of the used readout function
    def get_definition(self):
        return self.r_definition

    # Duvenaud
    def r_duvenaud(self, h):
        # layers
        aux = []
        for l in range(len(h)):
            param_sz = self.learn_args[l].size()
            parameter_mat = torch.t(self.learn_args[l])[None, ...].expand(h[l].size(0), param_sz[1],
                                                                                      param_sz[0])

            aux.append(torch.transpose(torch.bmm(parameter_mat, torch.transpose(h[l], 1, 2)), 1, 2))

            for j in range(0, aux[l].size(1)):
                # Mask whole 0 vectors
                aux[l][:, j, :] = nn.Softmax()(aux[l][:, j, :].clone())*(torch.sum(aux[l][:, j, :] != 0, 1) > 0)[...,None].expand_as(aux[l][:, j, :]).type_as(aux[l])

        aux = torch.sum(torch.sum(torch.stack(aux, 3), 3), 1)
        return self.learn_modules[0](torch.squeeze(aux))

    def init_duvenaud(self, params):
        learn_args = []
        learn_modules = []
        args = {}

        args['out'] = params['out']

        # Define a parameter matrix W for each layer.
        for l in range(params['layers']):
            learn_args.append(nn.Parameter(torch.randn(params['in'][l], params['out'])))

        # learn_modules.append(nn.Linear(params['out'], params['target']))

        learn_modules.append(NNet(n_in=params['out'], n_out=params['target']))
        return nn.ParameterList(learn_args), nn.ModuleList(learn_modules), args    
    
    
    # Battaglia et al. (2016), Interaction Networks
    def r_intnet(self, h):

        aux = torch.sum(h[-1],1)

        return self.learn_modules[0](aux)

    def init_intnet(self, params):
        learn_args = []
        learn_modules = []
        args = {}

        learn_modules.append(NNet(n_in=params['in'], n_out=params['target']))

        return nn.ParameterList(learn_args), nn.ModuleList(learn_modules), args

    def r_mpnn(self, h):

        aux = Variable( torch.Tensor(h[0].size(0), self.args['out']).type_as(h[0].data).zero_() )
        # For each graph
        for i in range(h[0].size(0)):
            nn_res = nn.Sigmoid()(self.learn_modules[0](torch.cat([h[0][i,:,:], h[-1][i,:,:]], 1)))*self.learn_modules[1](h[-1][i,:,:])

            # Delete virtual nodes
            nn_res = (torch.sum(h[0][i,:,:],1)[...,None].expand_as(nn_res)>0).type_as(nn_res)* nn_res

            aux[i,:] = torch.sum(nn_res,0)

        return aux

    def init_mpnn(self, params):
        learn_args = []
        learn_modules = []
        args = {}

        # i
        learn_modules.append(NNet(n_in=2*params['in'], n_out=params['target']))

        # j
        learn_modules.append(NNet(n_in=params['in'], n_out=params['target']))

        args['out'] = params['target']

        return nn.ParameterList(learn_args), nn.ModuleList(learn_modules), args

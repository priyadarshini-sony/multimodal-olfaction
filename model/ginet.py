import torch
from torch import nn
import torch.nn.functional as F

import argparse
import logging
import os
import random as rn
import sys
from collections import defaultdict

import numpy as np

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

num_atom_type = 119 # including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 5 # including aromatic and self-loop edge
num_bond_direction = 3 

class AutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_sizes,dropout):
        super(AutoEncoder, self).__init__()

        encoder_layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            encoder_layers.append(nn.Linear(prev_size, hidden_size))
            encoder_layers.append(nn.ReLU(inplace=True))
            encoder_layers.append(nn.Dropout(dropout))  
            prev_size = hidden_size

        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        prev_size = hidden_sizes[-1]
        for hidden_size in reversed(hidden_sizes[:-1]):
            decoder_layers.append(nn.Linear(prev_size, hidden_size))
            decoder_layers.append(nn.ReLU(inplace=True))
            decoder_layers.append(nn.Dropout(dropout)) 
            prev_size = hidden_size
        decoder_layers.append(nn.Linear(prev_size, input_size))

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded) 
        return decoded.float()

class GINEConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GINEConv, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2*emb_dim), 
            nn.ReLU(), 
            nn.Linear(2*emb_dim, emb_dim)
        )
        self.edge_embedding1 = nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, emb_dim)

        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + \
            self.edge_embedding2(edge_attr[:,1])

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GINet(nn.Module):
    """
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat
    Output:
        node representations
    """
    def __init__(self, 
        task='classification', num_layer=5, emb_dim=300, feat_dim=512, 
        drop_ratio=0, pool='mean', pred_n_layer=2, pred_act='softplus', out_dim=91
    ):
        super(GINet, self).__init__()
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.feat_dim = feat_dim
        self.drop_ratio = drop_ratio
        self.task = task

        self.x_embedding1 = nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = nn.Embedding(num_chirality_tag, emb_dim)
        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        # List of MLPs
        self.gnns = nn.ModuleList()
        for layer in range(num_layer):
            self.gnns.append(GINEConv(emb_dim))

        # List of batchnorms
        self.batch_norms = nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))

        if pool == 'mean':
            self.pool = global_mean_pool
        elif pool == 'max':
            self.pool = global_max_pool
        elif pool == 'add':
            self.pool = global_add_pool
        self.feat_lin = nn.Linear(self.emb_dim, self.feat_dim)
        
        self.pred_n_layer = max(1, pred_n_layer)

        if pred_act == 'relu':
            pred_head = [
                nn.Linear(self.feat_dim, self.feat_dim//2), 
                nn.ReLU(inplace=True)
            ]
            for _ in range(self.pred_n_layer - 1):
                pred_head.extend([
                    nn.Linear(self.feat_dim//2, self.feat_dim//2), 
                    nn.ReLU(inplace=True),
                ])
        elif pred_act == 'softplus':
            pred_head = [
                nn.Linear(self.feat_dim, self.feat_dim//2), 
                nn.Softplus()
            ]
            for _ in range(self.pred_n_layer - 1):
                pred_head.extend([
                    nn.Linear(self.feat_dim//2, self.feat_dim//2), 
                    nn.Softplus()
                ])
        else:
            raise ValueError('Undefined activation function')
        
        pred_head.append(nn.Linear(self.feat_dim//2, out_dim))
        pred_head.append(nn.Sigmoid())
        self.pred_head = nn.Sequential(*pred_head)

    def forward(self, data):
        x = data.x.to('cuda')
        edge_index = data.edge_index.to('cuda')
        edge_attr = data.edge_attr.to('cuda')
        
        h = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1])

        for layer in range(self.num_layer):
            h = self.gnns[layer](h, edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

        h = self.pool(h, data.batch.to('cuda'))
        
        h = self.feat_lin(h)
        return h, self.pred_head(h)

    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, nn.parameter.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)

class MOLCLREMBED_MLP(nn.Module):
    """
        MPNN as proposed by Gilmer et al..

        This class implements the whole Gilmer et al. model following the functions Message, Update and Readout.

        Parameters
        ----------
        in_n : int list
            Sizes for the node and edge features.
        hidden_state_size : int
            Size of the hidden states (the input will be padded with 0's to this size).
        message_size : int
            Message function output vector size.
        n_layers : int
            Number of iterations Message+Update (weight tying).
        l_target : int
            Size of the output.
        type : str (Optional)
            Classification | [Regression (default)]. If classification, LogSoftmax layer is applied to the output vector.
    """
    def __init__(self, l_target, embed_feature=512, final_op='CONCAT', dropout_embed=0.5):
        #super(MULTI, self).__init__()
        super().__init__()
        self.fc1_embed = torch.nn.Linear(embed_feature,512) 
        self.fc2_embed = torch.nn.Linear(512,128)
        self.fc3_embed = torch.nn.Linear(128,l_target)

        dropout_embed = 0.2

        self.dropout_embed = nn.Dropout(p=dropout_embed)
        
        self.ginet = GINet(task='classification', num_layer=5, emb_dim=300, feat_dim=512, 
                            drop_ratio=0.3, pool='mean', pred_n_layer=2, pred_act='softplus').to("cuda")

        checkpoints_folder = os.path.join('/home/ubuntu/work/multi-modal/ckpt', 'pretrained_gin', 'checkpoints')
        state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'), map_location='cuda')
        self.ginet.load_my_state_dict(state_dict)

        #concat, add, or element-wise multiply for 3 modalities
        self.final_op = final_op

        #different input size if add or multi
        self.last_fc = torch.nn.Linear(l_target*2,l_target)   

        if self.final_op == 'MUL' or self.final_op == 'ADD':
            self.last_fc = torch.nn.Linear(l_target,l_target)  

    def forward(self, batch_data, embeddings):
        # g, h_in, e, 
        
        #### transformer ####
        d_embed = F.relu(self.fc1_embed(embeddings))
        d_embed = F.relu(self.fc2_embed(d_embed))
        d_embed = torch.sigmoid(self.fc3_embed(d_embed))

        #### molclr ####
        _, d_molclr = self.ginet(batch_data)
        
        final_in = None
        if self.final_op == 'CONCAT':
            final_in = torch.cat((d_embed, d_molclr), 1)
        elif self.final_op == 'ADD':
            final_in = torch.add(d_embed, d_molclr)
        elif self.final_op == 'MUL':
            final_in = torch.mul(d_embed, d_molclr)

        concat_out = torch.sigmoid(self.last_fc(final_in))
        return concat_out

class MOLCLREMBED_LR(nn.Module):
    """
        MPNN as proposed by Gilmer et al..

        This class implements the whole Gilmer et al. model following the functions Message, Update and Readout.

        Parameters
        ----------
        in_n : int list
            Sizes for the node and edge features.
        hidden_state_size : int
            Size of the hidden states (the input will be padded with 0's to this size).
        message_size : int
            Message function output vector size.
        n_layers : int
            Number of iterations Message+Update (weight tying).
        l_target : int
            Size of the output.
        type : str (Optional)
            Classification | [Regression (default)]. If classification, LogSoftmax layer is applied to the output vector.
    """
    def __init__(self, embed_feature=512, dropout_embed=0.0, dropout=0.5):
        #super(MULTI, self).__init__()
        super().__init__()
        self.embed_bce = np.zeros(91)
        self.molclr_bce = np.zeros(91)

        #### transformer ####
        self.fc1_embed = torch.nn.Linear(embed_feature,512) 
        self.fc2_embed = torch.nn.Linear(512,128)
        self.fc3_embed = torch.nn.Linear(128,91)

        dropout_embed = 0.0
        self.dropout_embed = nn.Dropout(p=dropout_embed)
        
        self.ginet = GINet(task='classification', num_layer=5, emb_dim=300, feat_dim=512, 
                            drop_ratio=0.3, pool='mean', pred_n_layer=2, pred_act='softplus', out_dim=91).to("cuda")

        checkpoints_folder = os.path.join('/home/ubuntu/work/multi-modal/ckpt', 'pretrained_gin', 'checkpoints')
        state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'), map_location='cuda')
        self.ginet.load_my_state_dict(state_dict)

    def forward_test(self, batch_data, embeddings):
        # g, h_in, e, 
        d_embed = F.relu(self.fc1_embed(embeddings))
        d_embed = F.relu(self.fc2_embed(d_embed))
        d_embed = torch.sigmoid(F.relu(self.fc3_embed(d_embed)))

        #### molclr ####
        molclr_hidden, d_molclr = self.ginet(batch_data)
        final_tensor = torch.add(d_embed, d_molclr)
        final_tensor = final_tensor / 2.0

        return d_embed, d_molclr, final_tensor, molclr_hidden

    def forward(self, batch_data, embeddings):
        # g, h_in, e, 
        d_embed = F.relu(self.fc1_embed(embeddings))
        d_embed = F.relu(self.fc2_embed(d_embed))
        d_embed = torch.sigmoid(F.relu(self.fc3_embed(d_embed)))

        #### molclr ####
        _, d_molclr = self.ginet(batch_data)
        batch_size = d_embed.size()[0]

        import numpy as np
        length = 91
        mask = np.random.randint(2, size=length)

        embed_indices = np.where(mask == 1)[0]
        molclr_indices = np.where(mask == 0)[0]

        all_indices = np.concatenate((embed_indices, molclr_indices))

        final_tensor = torch.zeros(batch_size, 91).to('cuda')
        final_tensor[:, embed_indices] = d_embed[:, embed_indices]
        final_tensor[:, molclr_indices] = d_molclr[:, molclr_indices]

        final_tensor = final_tensor[:, all_indices]
        return all_indices, final_tensor
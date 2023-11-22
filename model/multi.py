import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import einsum
import random

random.seed(2022)


##Transformer 
class PreNorm(nn.Module): 
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout = 0.5):
        super().__init__()
        inner_dim = dim_head *  heads      
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim*3, bias = False)   ##better to dim to dim*3

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        h = self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        
        #v = k = q = self.to_qkv(x)
        
        #q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        #k = rearrange(k, 'b n (h d) -> b h n d', h=h)
        #v = rearrange(v, 'b n (h d) -> b h n d', h=h)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):           ##Register the blocks into whole network
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                #Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
                #Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                #FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

    
# MPNN    
    
# collate graph
def collate_g(batch):

    #batch_sizes = np.max(np.array([[len(input_b[1]), len(input_b[1][0]), 0,0] for (input_b, target_b) in batch]), axis=0)


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

class MULTI(nn.Module):
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
    def __init__(self, image_size,time_size,fre_size, dim, depth, heads, mlp_dim,dim_head,in_n, hidden_state_size, message_size, n_layers, numeric_feature, l_target, pool = 'cls' , type='regression', dropout = 0., emb_dropout = 0., final_op='CONCAT'):
        #super(MULTI, self).__init__()
        super().__init__()
        #### transformer ####
        num_patches = image_size      
        patch_dim = time_size * fre_size  
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        
        self.to_patch_embedding = nn.Linear(patch_dim, dim)      

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim)) ##Generate the pos value'
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))                ##Generate the class value'
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, l_target)

        self.sigmoid = nn.Sigmoid()
        
        
        # Define message
        self.m = nn.ModuleList(
            [MessageFunction('mpnn', args={'edge_feat': in_n[1], 'in': hidden_state_size, 
                                           'out': message_size})])

        # Define Update
        self.u = nn.ModuleList([UpdateFunction('mpnn',args={'in_m': message_size,
                                                            'out': hidden_state_size})])

        # Define Readout
        self.r = ReadoutFunction('mpnn',args={'in': hidden_state_size,
                                              'target': l_target})

        self.type = type

        self.args = {}
        self.args['out'] = hidden_state_size

        self.n_layers = n_layers
        
        #### mlp ####        
        self.fc1 = torch.nn.Linear(numeric_feature,512) 
        self.fc2 = torch.nn.Linear(512,128) 
        self.fc3 = torch.nn.Linear(128,l_target)   
        
        #### multi ####        
        self.numeric_feature = numeric_feature
        self.last_fc = torch.nn.Linear(l_target*3,l_target)   

        #concat, add, or element-wise multiply for 3 modalities
        self.final_op = final_op

        #different input size if add or multi
        if self.final_op == 'MUL' or self.final_op == 'ADD':
            self.last_fc = torch.nn.Linear(l_target,l_target)  

    def forward(self, g, h_in, e, din, img):
        # g, h_in, e, 

        #### MPNN ####
        h = []

        # Padding to some larger dimension d
        h_t = torch.cat([h_in, Variable(
            torch.zeros(h_in.size(0), h_in.size(1), self.args['out'] - h_in.size(2)).type_as(h_in.data))], 2)

        h.append(h_t.clone())

        # Layer
        for t in range(0, self.n_layers):
            e_aux = e.view(-1, e.size(3))

            h_aux = h[t].view(-1, h[t].size(2))

            m = self.m[0].forward(h[t], h_aux, e_aux)
            m = m.view(h[0].size(0), h[0].size(1), -1, m.size(1))

            # Nodes without edge set message to 0
            m = torch.unsqueeze(g, 3).expand_as(m) * m

            m = torch.squeeze(torch.sum(m, 1))

            h_t = self.u[0].forward(h[t], m)

            # Delete virtual nodes
            h_t = (torch.sum(h_in, 2)[..., None].expand_as(h_t) > 0).type_as(h_t) * h_t
            h.append(h_t)

        # Readout
        res = self.r.forward(h)
        # if self.type == 'classification':
        res = torch.sigmoid(res)

        dout = F.relu(self.fc1(din))  
        dout = F.relu(self.fc2(dout))
        dout = torch.sigmoid(self.fc3(dout))

        #### transformer ####
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        x = self.mlp_head(x)
        x = self.sigmoid(x)        
        
        final_in = None
        if self.final_op == 'CONCAT':
            final_in = torch.cat((res,dout,x), 1)
        elif self.final_op == 'ADD':
            final_in = torch.add(torch.add(res, dout), x)
        elif self.final_op == 'MUL':
            final_in = torch.mul(torch.mul(res, dout), x)

        concat_out = torch.sigmoid(self.last_fc(final_in))
        return concat_out

class MPNN_TRANS_LR(nn.Module):
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
    def __init__(self, image_size,time_size,fre_size, dim, depth, heads, mlp_dim,dim_head,in_n, hidden_state_size, message_size, n_layers, numeric_feature, l_target, pool = 'cls' , type='regression', dropout = 0., emb_dropout = 0., final_op='CONCAT'):
        #super(MULTI, self).__init__()
        super().__init__()
        #### transformer ####
        num_patches = image_size      
        patch_dim = time_size * fre_size  
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        
        self.to_patch_embedding = nn.Linear(patch_dim, dim)      

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim)) ##Generate the pos value'
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))                ##Generate the class value'
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, l_target)

        self.sigmoid = nn.Sigmoid()
        
        
        # Define message
        self.m = nn.ModuleList(
            [MessageFunction('mpnn', args={'edge_feat': in_n[1], 'in': hidden_state_size, 
                                           'out': message_size})])

        # Define Update
        self.u = nn.ModuleList([UpdateFunction('mpnn',args={'in_m': message_size,
                                                            'out': hidden_state_size})])

        # Define Readout
        self.r = ReadoutFunction('mpnn',args={'in': hidden_state_size,
                                              'target': l_target})

        self.type = type

        self.args = {}
        self.args['out'] = hidden_state_size

        self.n_layers = n_layers
        
        #### multi ####        
        self.numeric_feature = numeric_feature
        self.last_fc = torch.nn.Linear(l_target*2,l_target)   

        #concat, add, or element-wise multiply for 3 modalities
        self.final_op = final_op

        #different input size if add or multi
        if self.final_op == 'MUL' or self.final_op == 'ADD':
            self.last_fc = torch.nn.Linear(l_target,l_target)

    def forward_test(self, g, h_in, e, din, img):
        # g, h_in, e, 

        #### MPNN ####
        h = []

        # Padding to some larger dimension d
        h_t = torch.cat([h_in, Variable(
            torch.zeros(h_in.size(0), h_in.size(1), self.args['out'] - h_in.size(2)).type_as(h_in.data))], 2)

        h.append(h_t.clone())

        # Layer
        for t in range(0, self.n_layers):
            e_aux = e.view(-1, e.size(3))

            h_aux = h[t].view(-1, h[t].size(2))

            m = self.m[0].forward(h[t], h_aux, e_aux)
            m = m.view(h[0].size(0), h[0].size(1), -1, m.size(1))

            # Nodes without edge set message to 0
            m = torch.unsqueeze(g, 3).expand_as(m) * m

            m = torch.squeeze(torch.sum(m, 1))

            h_t = self.u[0].forward(h[t], m)

            # Delete virtual nodes
            h_t = (torch.sum(h_in, 2)[..., None].expand_as(h_t) > 0).type_as(h_t) * h_t
            h.append(h_t)

        # Readout
        res = self.r.forward(h)
        # if self.type == 'classification':
        res = torch.sigmoid(res)

        #### transformer ####
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        x = self.mlp_head(x)
        x = self.sigmoid(x)

        final_tensor = torch.add(res, x)
        final_tensor = final_tensor / 2.0

        return final_tensor

    def forward(self, g, h_in, e, din, img):
        # g, h_in, e, 

        #### MPNN ####
        h = []

        # Padding to some larger dimension d
        h_t = torch.cat([h_in, Variable(
            torch.zeros(h_in.size(0), h_in.size(1), self.args['out'] - h_in.size(2)).type_as(h_in.data))], 2)

        h.append(h_t.clone())

        # Layer
        for t in range(0, self.n_layers):
            e_aux = e.view(-1, e.size(3))

            h_aux = h[t].view(-1, h[t].size(2))

            m = self.m[0].forward(h[t], h_aux, e_aux)
            m = m.view(h[0].size(0), h[0].size(1), -1, m.size(1))

            # Nodes without edge set message to 0
            m = torch.unsqueeze(g, 3).expand_as(m) * m

            m = torch.squeeze(torch.sum(m, 1))

            h_t = self.u[0].forward(h[t], m)

            # Delete virtual nodes
            h_t = (torch.sum(h_in, 2)[..., None].expand_as(h_t) > 0).type_as(h_t) * h_t
            h.append(h_t)

        # Readout
        res = self.r.forward(h)
        res = torch.sigmoid(res)

        #### transformer ####
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        x = self.mlp_head(x)
        x = self.sigmoid(x)

        batch_size = x.size()[0]

        import numpy as np
        length = 91
        mask = np.random.randint(2, size=length)

        x_indices = np.where(mask == 1)[0]
        res_indices = np.where(mask == 0)[0]

        all_indices = np.concatenate((x_indices, res_indices))

        final_tensor = torch.zeros(batch_size, 91).to('cuda')
        final_tensor[:, x_indices] = x[:, x_indices]
        final_tensor[:, res_indices] = res[:, res_indices]

        return final_tensor 
        
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.nn.parameter import Parameter



class DeconvNet1(nn.Module):

    def __init__(self, 
                 hidden_dims, # dimensionality of hidden layers
                 n_celltypes, # number of cell types
                 n_slices, # number of slices
                 n_heads, # number of attention heads
                 slice_emb_dim, # dimensionality of slice id embedding
                 coef_fe,
                 ):

        super().__init__()

        # define layers
        # encoder layers
        self.encoder_layer1 = GATMultiHead(hidden_dims[0], hidden_dims[1], n_heads=n_heads, concat_heads=True)
        self.encoder_layer2 = DenseLayer(hidden_dims[1], hidden_dims[2])
        # decoder layers
        self.decoder_layer1 = GATMultiHead(hidden_dims[2] + slice_emb_dim, hidden_dims[1], n_heads=n_heads, concat_heads=True)
        self.decoder_layer2 = DenseLayer(hidden_dims[1], hidden_dims[0])



        self.slice_emb = nn.Embedding(n_slices, slice_emb_dim)

        self.coef_fe = coef_fe

        self.fc = nn.Linear(33,7)

    def forward(self, 
                adj_matrix, # adjacency matrix including self-connections
                node_feats, # input node features
                count_matrix, # gene expression counts
                library_size, # library size (based on Y)
                slice_label, # slice label
                basis, # basis matrix
                ):
        # encoder
        Z = self.encoder(adj_matrix, node_feats)

        # deconvolutioner
        slice_label_emb = self.slice_emb(slice_label)


        # decoder
        node_feats_recon = self.decoder(adj_matrix, Z, slice_label_emb)

        return node_feats_recon, Z

    def evaluate(self, adj_matrix, node_feats, slice_label):
        slice_label_emb = self.slice_emb(slice_label)
        # encoder
        Z = self.encoder(adj_matrix, node_feats)
        



        return Z#, beta, alpha, self.gamma
            
    def encoder(self, adj_matrix, node_feats):
        H = node_feats
        H = F.elu(self.encoder_layer1(H, adj_matrix))
        Z = self.encoder_layer2(H)
        return Z
        
    def decoder(self, adj_matrix, Z, slice_label_emb):
        H = torch.cat((Z, slice_label_emb), axis=1)
        H = F.elu(self.decoder_layer1(H, adj_matrix))
        X_recon = self.decoder_layer2(H)
        return X_recon
       
class Encoder_map(torch.nn.Module):
    def __init__(self, n_cell, n_spot):
        super(Encoder_map, self).__init__()
        self.n_cell = n_cell
        self.n_spot = n_spot
          
        self.M = Parameter(torch.FloatTensor(self.n_cell, self.n_spot))
        # self.PyConv = 
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.M)
        
    def forward(self):
        x = self.M
        
        return x 
    

class GATMultiHead(nn.Module):

    def __init__(self, 
                 c_in, # dimensionality of input features
                 c_out, # dimensionality of output features
    	         n_heads=1, # number of attention heads
    	         concat_heads=True, # concatenate attention heads or not
    	         ):

        super().__init__()

        self.n_heads = n_heads
        self.concat_heads = concat_heads
        if self.concat_heads:
            assert c_out % n_heads == 0, "The number of output features should be divisible by the number of heads."
            c_out = c_out // n_heads

        self.block = nn.ModuleList()
        for i_block in range(self.n_heads):
            self.block.append(GATSingleHead(c_in=c_in, c_out=c_out))

    def forward(self, 
                node_feats, # input node features
                adj_matrix, # adjacency matrix including self-connections
                ):

        res = []
        for i_block in range(self.n_heads):
            res.append(self.block[i_block](node_feats, adj_matrix))
        
        if self.concat_heads:
            node_feats = torch.cat(res, dim=1)
        else:
            node_feats = torch.mean(torch.stack(res, dim=0), dim=0)

        return node_feats




class GATSingleHead(nn.Module):

    def __init__(self, 
                 c_in, # dimensionality of input features
                 c_out, # dimensionality of output features
                 temp=1, # temperature parameter
                 ):

        super().__init__()

        self.linear = nn.Linear(c_in, c_out)
        self.v0 = nn.Parameter(torch.Tensor(c_out, 1))
        self.v1 = nn.Parameter(torch.Tensor(c_out, 1))
        self.temp = temp

        # Initialization
        nn.init.uniform_(self.linear.weight.data, -np.sqrt(6 / (c_in + c_out)), np.sqrt(6 / (c_in + c_out)))
        nn.init.zeros_(self.linear.bias.data)
        nn.init.uniform_(self.v0.data, -np.sqrt(6 / (c_out + 1)), np.sqrt(6 / (c_out + 1)))
        nn.init.uniform_(self.v1.data, -np.sqrt(6 / (c_out + 1)), np.sqrt(6 / (c_out + 1)))

    def forward(self, 
                node_feats, # input node features
                adj_matrix, # adjacency matrix including self-connections
                ):

        # Apply linear layer and sort nodes by head
        node_feats = self.linear(node_feats)
        f1 = torch.matmul(node_feats, self.v0)
        f2 = torch.matmul(node_feats, self.v1)
        attn_logits = adj_matrix * (f1 + f2.T)
        unnormalized_attentions = (F.sigmoid(attn_logits) - 0.5).to_sparse()
        attn_probs = torch.sparse.softmax(unnormalized_attentions / self.temp, dim=1)
        attn_probs = attn_probs.to_dense()
        node_feats = torch.matmul(attn_probs, node_feats)

        return node_feats
    


class DenseLayer(nn.Module):

    def __init__(self, 
                 c_in, # dimensionality of input features
                 c_out, # dimensionality of output features
                 zero_init=False, # initialize weights as zeros; use Xavier uniform init if zero_init=False
                 ):

        super().__init__()

        self.linear = nn.Linear(c_in, c_out)

        # Initialization
        if zero_init:
            nn.init.zeros_(self.linear.weight.data)
        else:
            nn.init.uniform_(self.linear.weight.data, -np.sqrt(6 / (c_in + c_out)), np.sqrt(6 / (c_in + c_out)))
        nn.init.zeros_(self.linear.bias.data)

    def forward(self, 
    			node_feats, # input node features
    			):

        node_feats = self.linear(node_feats)

        return node_feats

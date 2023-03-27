
import os.path as osp

import torch
import torch.nn as nn
from tqdm import tqdm

from torch_geometric.datasets import Reddit
from torch_geometric.loader import NeighborSampler
from torch_geometric.nn import DeepGraphInfomax, SAGEConv, GATv2Conv,GATConv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DeepGraphInfomax, GCNConv


import scanpy as sc
import networkx as nx
import pandas as pd
import pickle
import os.path as osp

import torch
import torch.nn as nn
from tqdm import tqdm


from torch_geometric.loader import NeighborSampler
from torch_geometric.nn import DeepGraphInfomax, SAGEConv, GATv2Conv,GATConv
import torch
import torch.nn as nn
import torch.nn.functional as F
import anndata as ad
from torch_geometric.nn import DeepGraphInfomax, GCNConv
from torch.nn import Linear as Lin



class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        headz = 3
        hidden_channelsz = 256
        self.conv = GATv2Conv(in_channels, hidden_channelsz,heads =headz)
        #self.conv = GCNConv(in_channels, hidden_channels,heads =headz )
        self.conv1 = GATv2Conv(hidden_channels*headz, hidden_channelsz,heads =headz )
        self.conv2 = GATv2Conv(hidden_channelsz*headz, hidden_channels,heads =headz,concat =False,dropout =0.1)#,dropout =0.4

        self.prelu = nn.PReLU(hidden_channelsz*headz)
        self.prelu1 = nn.PReLU(hidden_channels)

        self.prelu2 = nn.PReLU(hidden_channelsz*headz)

    def forward(self, x, edge_index):
        
        x = self.conv(x, edge_index)
        x = self.prelu(x)
        #x,attn1 = self.conv1(x,edge_index,return_attention_weights=True)
        #x = self.prelu2(x)
        x,attn2 = self.conv2(x,edge_index,return_attention_weights=True)#
        x = self.prelu1(x)
        return x,attn2#,attn1


class EncoderT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, heads):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads, dropout=0.6)

        self.conv12 = GATv2Conv(hidden_channels*heads, hidden_channels, heads, dropout=0.6)

        # On the Pubmed dataset, use `heads` output heads in `conv2`.
        self.conv2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=1,
                             concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))

        x = F.dropout(x, p=0.6, training=self.training)
        x,attn2 = self.conv12(x, edge_index,return_attention_weights=True)
        x = F.elu(x)

        x = F.dropout(x, p=0.6, training=self.training)
        x,attn3 = self.conv2(x, edge_index,return_attention_weights=True)
        return x,attn2,attn3



class EncoderT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers,
                 heads,dropout = True):
        super().__init__()

        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(GATv2Conv(in_channels, hidden_channels,
                                  heads,share_weights=True))
        for _ in range(num_layers - 2):
            self.convs.append(
                GATv2Conv(heads * hidden_channels, hidden_channels, heads,share_weights=True))
        self.convs.append(
            GATv2Conv(heads * hidden_channels, hidden_channels, heads,share_weights=True,
                    concat=False))

        self.skips = torch.nn.ModuleList()
        self.skips.append(Lin(in_channels, hidden_channels * heads))
        for _ in range(num_layers - 2):
            self.skips.append(
                Lin(hidden_channels * heads, hidden_channels * heads))
        self.skips.append(Lin(hidden_channels * heads, hidden_channels))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for skip in self.skips:
            skip.reset_parameters()
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for skip in self.skips:
            skip.reset_parameters()

    def forward(self, x, edge_index):
        # `train_loader` computes the k-hop neighborhood of a batch of nodes,
        # and returns, for each layer, a bipartite graph object, holding the
        # bipartite edges `edge_index`, the index `e_id` of the original edges,
        # and the size/shape `size` of the bipartite graph.
        # Target nodes are also included in the source nodes so that one can
        # easily apply skip-connections or add self-loops.
        attention = [tuple() for i in range(self.num_layers)]
        for i in range(self.num_layers):
              # Target nodes are always placed first.
            #print(i)
            #print(x.size())
            if dropout == True:
                if i != self.num_layers - 1:
            	     x = F.dropout(x, p=0.5, training=self.training)
            x,attz = self.convs[i](x, edge_index,return_attention_weights = True)
            #print("ok")
            attention[i] = attention[i] + attz
            #x = x + self.skips[i](x)
            #print("no way")
            #print(x.size())
            if i != self.num_layers - 1:
                x = F.elu(x)
                #x = F.dropout(x, p=0.5, training=self.training)
            
                
        self.attention = attention
        return x,1


#HVG,hidden,Elayer,attn
class DeepGInfomaxT():
    
    def __init__(self,G,ep,hidden,HVG,Elayer,attn,verbose = True,dropout ==False):
    
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = EncoderT(HVG,hidden,Elayer,attn,dropout)
        self.model = DeepGraphInfomax(
                hidden_channels=hidden, encoder=self.encoder,
                summary=lambda z, *args, **kwargs: torch.sigmoid(z[0].mean(dim=0)),
                corruption=self.corruption).to(device)
        self.data = G.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        
        self.history=[]
        for epoch in range(1, ep):
            loss = self.train()
            self.history.append(loss)
            if verbose ==True:
            	print(f'Epoch: {epoch:03d}, Loss: {loss:.7f}')


         


    def corruption(self,x, edge_index):
        return x[torch.randperm(x.size(0))], edge_index




    def train(self):
        self.model.train()
        self.optimizer.zero_grad()
        pos_z, neg_z, summary = self.model(self.data.x, self.data.edge_index)
        loss = self.model.loss(pos_z[0], neg_z[0], summary)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()


    def test(self):
        self.model.eval()
        z, _, _ = self.model(self.data.x, self.data.edge_index)
        #acc = model.test(z[data.train_mask], data.y[data.train_mask],
        #                z[data.test_mask], data.y[data.test_mask], max_iter=150)
        return z
        
    def EmbAttention(self):
        with torch.no_grad():
            z = self.test()
        
        return z


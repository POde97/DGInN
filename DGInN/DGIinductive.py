
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

class EncoderI(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers,
                 heads):
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

    def forward(self, x, adjs):
        # `train_loader` computes the k-hop neighborhood of a batch of nodes,
        # and returns, for each layer, a bipartite graph object, holding the
        # bipartite edges `edge_index`, the index `e_id` of the original edges,
        # and the size/shape `size` of the bipartite graph.
        # Target nodes are also included in the source nodes so that one can
        # easily apply skip-connections or add self-loops.
        attention = [tuple() for j in range(len(self.convs))]
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x,attz = self.convs[i]((x, x_target), edge_index,return_attention_weights = True)
            attention[i] = attention[i] + attz
            x = x + self.skips[i](x_target)
            if i != self.num_layers - 1:
                x = F.elu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        self.attention = attention 

        return x,1


class DeepGInfomaxI():

  def __init__(self,G,ep,hidden = 256,HVG = 4000,Elayer = 3,attn = 4,siz=[5,10,15],batch = 512,patience=3):
    
    #Data Loader
    self.data = G
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.train_loader = NeighborSampler(self.data.edge_index, node_idx=None,
                               sizes=siz, batch_size=batch,
                               shuffle=True, num_workers=6)

    self.test_loader = NeighborSampler(self.data.edge_index, node_idx=None,
                              sizes=siz, batch_size=batch,
                              shuffle=False, num_workers=6)



    self.encoder = EncoderI(HVG, hidden,Elayer,attn)
    self.model = DeepGraphInfomax(
      hidden_channels=hidden, encoder=self.encoder,
      summary=lambda z, *args, **kwargs: torch.sigmoid(z[0].mean(dim=0)),
      corruption=self.corruption).to(self.device)

    self.model = self.model.to(self.device)
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)

    self.x = self.data.feature.to(self.device)


    last_loss = 100
    #patience = 
    triggertimes = 0
    listofmodel=[]
    self.LL=[]

    for epoch in range(1,ep):

      T = self.train(epoch)
      current_loss = T[0]
      self.LL.append(current_loss)
      if current_loss > last_loss:
        trigger_times += 1
        print('Trigger Times:', trigger_times)
        listofmodel.append(T[2])

        if trigger_times >= patience:
          print('Early stopping!\nStart to test process.')
        #model = listofmodel[0]
          break
    
      else:
        print('trigger times: 0')
        trigger_times = 0

      
      last_loss = current_loss


      print(f'Epoch {epoch:02d}, Loss: {current_loss:.8f}')


  def corruption(self,x, edge_index):
    return x[torch.randperm(x.size(0))], edge_index


  def train(self,epoch):
    self.model.train()

    total_loss = total_examples = 0
    for batch_size, n_id, adjs in tqdm(self.train_loader,
                                       desc=f'Epoch {epoch:02d}'):
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs = [adj.to(self.device) for adj in adjs]

        self.optimizer.zero_grad()
        pos_z, neg_z, summary = self.model(self.x[n_id], adjs)
        loss = self.model.loss(pos_z[0], neg_z[0], summary)
        loss.backward()
        self.optimizer.step()
        total_loss += float(loss) * pos_z[0].size(0)
        total_examples += pos_z[0].size(0)

    return total_loss / total_examples , pos_z[1],self.model



  
  def test(self):
    self.model.eval()

    zs = []
    for i, (batch_size, n_id, adjs) in enumerate(self.test_loader):
        adjs = [adj.to(self.device) for adj in adjs]
        zs.append(self.model(self.x[n_id], adjs)[0][0])
    z = torch.cat(zs, dim=0)
    
    return z


  
  def EmbAttention(self):
    with torch.no_grad():
    
      z = self.test()
    return z






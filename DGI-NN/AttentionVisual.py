#Attention Visualization function 
#wrapped by CellVGAE Buterez et al https://github.com/davidbuterez/CellVGAE
import numpy as np 
import torch
from scipy.stats import entropy
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import igraph as ig
import seaborn as sns
from collections import defaultdict
import scanpy as sc


def get_cell_weights_for_layer(layer_edge_index, layer_coeff):
    cell_weights = []
    for dim in range(0, ATTN_HEADS):
        cell_weights_dim = []
        for cell_id in range(num_cells):
            dim0 = layer_edge_index[0] == cell_id
            dim1 = layer_edge_index[1] == cell_id
            idxs = torch.logical_or(dim0, dim1)
            #print(idxs,dim1,dim0)
            cell_weight = torch.sum(layer_coeff[idxs, dim])
            cell_weights_dim.append(cell_weight.detach().numpy())
        cell_weights.append(cell_weights_dim)
    return np.array(cell_weights)


def extract_attn_data(edge_index_attn, weights_attn, dim=None, k=80):
    edges = edge_index_attn.T
    if not dim:
        w = weights_attn.mean(dim=1)
    else:
        w = weights_attn[:, dim]
    w = w.squeeze()
    top_values, top_indices = torch.topk(w, k)
    top_edges = edges[top_indices]
    
    return top_edges, top_values
    
def HentopyDistribution(G,headz):
    #Hentropy
    df = nx.to_pandas_edgelist(G1)
    head = headz
    ll = []
    for i in list(df["source"].unique()):
        w = np.array(list(df.loc[df["source"]== i]["weight"].to_numpy()))[:,head]
        H = entropy(list(w))
        ll.append(H)

    plt.hist(ll,bins=10)
    
    
class AttentionNet():
  def __init__(self,adata,cell_key,edge_index,attn_coeff,ktop):
    self.adata = adata
    self.key = cell_key
    #Rename Cell
    new_name = self.rename()

    edge_index, edge_weights = self.extract_attn_data(edge_index, attn_coeff, k=ktop)
    edge_index = edge_index.numpy()
    edge_weights = edge_weights.detach().numpy()
    
    #Build Network Top K edges
    g = ig.Graph(edges=edge_index, directed=True)
    g.es['weight'] = edge_weights
    g.vs['name1'] = new_name
    g_selected = g.subgraph(g.vs.select(lambda vertex: vertex.degree() > 0))
    
    self.G = g_selected.to_networkx()






  def rename(self):
    seurat_df = self.adata.obs
    cell_types = seurat_df[self.key].values.tolist()
    num_cells = len(cell_types)
    cell_to_cluster = seurat_df.to_dict()[self.key]
    cell_type_counter = defaultdict(int)
    new_cell_clusters = []
    for cell_cluster in cell_to_cluster.values():
      cnt = cell_type_counter[cell_cluster]
      cell_type_counter[cell_cluster] += 1
      new_cell_clusters.append(cell_cluster + '--' + str(cnt))
    return new_cell_clusters

  def extract_attn_data(self,edge_index_attn, weights_attn, dim=None, k=120):
    edges = edge_index_attn.T
    if not dim:
        w = weights_attn.mean(dim=1)
    else:
        w = weights_attn[:, dim]
    w = w.squeeze()
    top_values, top_indices = torch.topk(w, k)
    top_edges = edges[top_indices]
    
    return top_edges, top_values
    
    
def PlotAttentionG(G1):
  
  #Plot Network 
  tt = dict(G1.nodes(data=True))
  cell_type = list(np.unique([tt[i]["name1"].split("--")[0] for i in range(len(list(tt.keys())))]))
  cmap = ["blue","grey","green","yellow","orange","red","purple","brown","lime","seagreen",
          "plum","paleviolettred","m","aliceblue","slategray","tan","floralwhite","sandybrown",
          "rosybrown","gold","olivedrab"]
  cmap = cmap[:len(cell_type)]
  color_map = []
  for node in G1:

    color_temp_idx = cell_type.index(G1.nodes(data=True)[node]["name1"].split("--")[0])
    color_map.append(cmap[color_temp_idx])

  plt.figure(figsize=(10,10))
  weightz = nx.get_edge_attributes(G1,'weight').values()  
  nx.draw(G1,pos = nx.nx_pydot.pydot_layout(G1),node_color = color_map , node_size = 50,width=list(weightz))
  
  #Plot legend
  for v in range(len(cmap)):
    plt.scatter([],[], c=cmap[v], label=cell_type[v])

  plt.legend()
  plt.show()

    
    
    
    
    
    



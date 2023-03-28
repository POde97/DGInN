import networkx as nx
import scanpy as sc
import pandas as pd
import scipy


def SetNodeAttribute_G(G,adata,scaler=None,batch_key = "batch",inductive_transductive="inductive",HVG=4000):

  adata.raw = adata#.copy()
  sc.pp.normalize_total(adata, target_sum=1e4)
  sc.pp.log1p(adata)
  adata.obs[batch_key] = adata.obs[batch_key].astype('category')
  sc.pp.highly_variable_genes(adata,n_top_genes=HVG,batch_key=batch_key)
  
  del adata.uns["log1p"]
  adata.X = adata.raw.X
  adata = adata[:, adata.var.highly_variable].copy()
  sc.pp.log1p(adata)

  

  from sklearn.preprocessing import StandardScaler
  if scipy.sparse.issparse(adata.X)== True:
    scaler = StandardScaler().fit(np.array(adata.X.todense()))
    X_scaler = scaler.transform(np.array(adata.X.todense()))
  else:
    scaler = StandardScaler().fit(np.array(adata.X))
    X_scaler = scaler.transform(np.array(adata.X))

  Hvar_final = pd.DataFrame(X_scaler,columns = list(adata.var_names))
  Hvar_final.index = adata.obs_names
  Hvar_final = Hvar_final[Hvar_final.index.isin(list(G.nodes))]
    
  batchdf = pd.DataFrame(adata.obs[batch_key])
  batchdf = batchdf.loc[list(G.nodes)]
  #Add genes as features
  if inductive_transductive =="inductive":
    nx.set_node_attributes(G,dict(zip(list(Hvar_final.index),Hvar_final.dropna(axis=1).values.tolist())), name="feature")
  else:
    nx.set_node_attributes(G,dict(zip(list(Hvar_final.index),Hvar_final.dropna(axis=1).values.tolist())), name="feature")
  
  #Add batch as features
  nx.set_node_attributes(G, dict(zip(list(batchdf.index),list(batchdf[batch_key]))) , name="node_type")


  return G



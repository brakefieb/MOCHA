import numpy as np
import scanpy as sc
import anndata as ad
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from matplotlib import cm
import torch.nn.functional as F
import random
seed = 2024
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



def select_hvgs(adata_ref, celltype_ref_col, num_per_group=200):
    sc.tl.rank_genes_groups(adata_ref, groupby=celltype_ref_col, method="t-test", key_added="ttest", use_raw=False)
    markers_df = pd.DataFrame(adata_ref.uns['ttest']['names']).iloc[0:num_per_group, :]
    genes = sorted(list(np.unique(markers_df.melt().value.values)))
    return genes


def calculate_impubasis(adata_st_input, #st anndata object (should be one of the output from STitch3D.utils.preprocess)
                        adata_ref_input, # reference single-cell anndata object (raw data)
                        celltype_ref_col="celltype", # column of adata_ref_input.obs for cell type information
                        sample_col=None, # column of adata_ref_input.obs for batch labels
                        celltype_ref=None, # specify cell types to use for deconvolution
                        ):
    
    adata_ref = adata_ref_input.copy()
    adata_ref.var_names_make_unique()
    # Remove mt-genes
    adata_ref = adata_ref[:, np.array(~adata_ref.var.index.isna())
                          & np.array(~adata_ref.var_names.str.startswith("mt-"))
                          & np.array(~adata_ref.var_names.str.startswith("MT-"))]
    if celltype_ref is not None:
        if not isinstance(celltype_ref, list):
            raise ValueError("'celltype_ref' must be a list!")
        else:
            adata_ref = adata_ref[[(t in celltype_ref) for t in adata_ref.obs[celltype_ref_col].values.astype(str)], :]
    else:
        celltype_counts = adata_ref.obs[celltype_ref_col].value_counts()
        celltype_ref = list(celltype_counts.index[celltype_counts > 1])
        adata_ref = adata_ref[[(t in celltype_ref) for t in adata_ref.obs[celltype_ref_col].values.astype(str)], :]

    # Remove cells and genes with 0 counts
    sc.pp.filter_cells(adata_ref, min_genes=1)
    sc.pp.filter_genes(adata_ref, min_cells=1)

    # Calculate single cell library sizes
    hvgs = adata_st_input.var.index
    adata_ref_ls = adata_ref[:, hvgs]
    sc.pp.filter_cells(adata_ref_ls, min_genes=1)
    adata_ref = adata_ref[adata_ref_ls.obs.index, :]
    # ref_ls: library size (only account for hvgs) of single cells in ref
    if scipy.sparse.issparse(adata_ref_ls.X):
        ref_ls = np.sum(adata_ref_ls.X.toarray(), axis=1).reshape((-1,1))
        adata_ref.obsm["forimpu"] = adata_ref.X.toarray() / ref_ls
    else:
        ref_ls = np.sum(adata_ref_ls.X, axis=1).reshape((-1,1))
        adata_ref.obsm["forimpu"] = adata_ref.X / ref_ls

    # Calculate basis for imputation
    celltype_list = list(sorted(set(adata_ref.obs[celltype_ref_col].values.astype(str))))
    basis_impu = np.zeros((len(celltype_list), len(adata_ref.var.index)))
    if sample_col is not None:
        sample_list = list(sorted(set(adata_ref.obs[sample_col].values.astype(str))))
        for i in range(len(celltype_list)):
            c = celltype_list[i]
            tmp_list = []
            for j in range(len(sample_list)):
                s = sample_list[j]
                tmp = adata_ref[(adata_ref.obs[celltype_ref_col].values.astype(str) == c) & 
                                (adata_ref.obs[sample_col].values.astype(str) == s), :].obsm["forimpu"]
                if scipy.sparse.issparse(tmp):
                    tmp = tmp.toarray()
                if tmp.shape[0] >= 3:
                    tmp_list.append(np.mean(tmp, axis=0).reshape((-1)))
            tmp_mean = np.mean(tmp_list, axis=0)
            if scipy.sparse.issparse(tmp_mean):
                tmp_mean = tmp_mean.toarray()
            print("%d batches are used for computing the basis vector of cell type <%s>." % (len(tmp_list), c))
            basis_impu[i, :] = tmp_mean
    else:
        for i in range(len(celltype_list)):
            c = celltype_list[i]
            tmp = adata_ref[adata_ref.obs[celltype_ref_col].values.astype(str) == c, :].obsm["forimpu"]
            if scipy.sparse.issparse(tmp):
                tmp = tmp.toarray()
            basis_impu[i, :] = np.mean(tmp, axis=0).reshape((-1))

    adata_basis_impu = ad.AnnData(X=basis_impu)
    df_gene = pd.DataFrame({"gene": adata_ref.var.index})
    df_gene = df_gene.set_index("gene")
    df_celltype = pd.DataFrame({"celltype": celltype_list})
    df_celltype = df_celltype.set_index("celltype")
    adata_basis_impu.obs = df_celltype
    adata_basis_impu.var = df_gene
    adata_basis_impu = adata_basis_impu[~np.isnan(adata_basis_impu.X[:, 0])]
    return adata_basis_impu

def Noise_Cross_Entropy(adata_st, pred_sp, emb_sp, device):
        '''\
        Calculate noise cross entropy. Considering spatial neighbors as positive pairs for each spot
            
        Parameters
        ----------
        pred_sp : torch tensor
            Predicted spatial gene expression matrix.
        emb_sp : torch tensor
            Reconstructed spatial gene expression matrix.

        Returns
        -------
        loss : float
            Loss value.

        '''
        
        mat = cosine_similarity(pred_sp, emb_sp) 
        k = torch.exp(mat).sum(axis=1) - torch.exp(torch.diag(mat, 0))
        
        # positive pairs
        p = torch.exp(mat)
        graph_neigh = torch.FloatTensor(adata_st.obsm['graph'].copy() + np.eye(adata_st.obsm['graph'].shape[0])).to(device)
        p = torch.mul(p, graph_neigh).sum(axis=1)
        
        ave = torch.div(p, k)
        loss = - torch.log(ave).mean()
        
        return loss


    

def cosine_similarity(pred_sp, emb_sp):  #pres_sp: spot x gene; emb_sp: spot x gene
    '''\
    Calculate cosine similarity based on predicted and reconstructed gene expression matrix.    
    '''
    
    M = torch.matmul(pred_sp, emb_sp.T)
    Norm_c = torch.norm(pred_sp, p=2, dim=1)
    Norm_s = torch.norm(emb_sp, p=2, dim=1)
    Norm = torch.matmul(Norm_c.reshape((pred_sp.shape[0], 1)), Norm_s.reshape((emb_sp.shape[0], 1)).T) + -5e-12
    M = torch.div(M, Norm)
    
    if torch.any(torch.isnan(M)):
        M = torch.where(torch.isnan(M), torch.full_like(M, 0.4868), M)

    return M        


def LL_loss(adata_st, emb_sp, emb_sc, map_matrix, device):
    '''\
    Calculate loss

    Parameters
    ----------
    emb_sp : torch tensor
        Spatial spot representation matrix.
    emb_sc : torch tensor
        scRNA cell representation matrix.

    Returns
    -------
    Loss values.

    '''
    # cell-to-spot
    map_probs = F.softmax(map_matrix, dim=1)   # dim=0: normalization by cell
    pred_sp = torch.matmul(map_probs.t(), emb_sc)
        
    loss_recon = F.mse_loss(pred_sp, emb_sp, reduction='mean')
    loss_NCE = Noise_Cross_Entropy(adata_st, pred_sp, emb_sp, device)
    # loss_NCE = Noise_Cross_Entropy(pred_sp, emb_sp)
    return loss_recon, loss_NCE

def find_resolution(adata_, n_clusters, random_state): 
    adata = adata_.copy()
    obtained_clusters = -1
    iteration = 0
    resolutions = [0., 1000.]
    while obtained_clusters != n_clusters and iteration < 50:
        current_res = sum(resolutions)/2
        sc.tl.louvain(adata, resolution=current_res, random_state=random_state)
        labels = adata.obs['louvain']
        obtained_clusters = len(np.unique(labels))

        if obtained_clusters < n_clusters:
            resolutions[0] = current_res
        else:
            resolutions[1] = current_res
        iteration = iteration + 1
    final_cluster=len(np.unique(adata.obs['louvain']))

    return current_res


def set_seed(seed: int = 2024):
    """
    Set random seed for reproducibility across random, numpy, and torch.

    Parameters
    ----------
    seed : int
        The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def construct_combined_graph(adata_st, loc_3D, map_path, bl=0.1, bll=0.1, radius=150.7):
    d_geo = pairwise_distances(adata_st.obsm["3D_coor"], metric="euclidean")
    d_map = pairwise_distances(np.load(map_path))
    d_img = pairwise_distances(loc_3D)
    d_combined = d_geo + d_map * bl + d_img * bll
    G = (d_combined < radius).astype(float)
    return G
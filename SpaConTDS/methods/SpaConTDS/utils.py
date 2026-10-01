import argparse
import anndata
import os, cv2
import numpy as np
import pandas as pd
import scanpy as sc
# import squidpy as sq
import sklearn.neighbors
from scipy.spatial.distance import cosine
from scipy import sparse as scipy_sparse

import ot
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import subgraph as pyg_subgraph
import torch.backends.cudnn as cudnn
cudnn.deterministic = True
cudnn.benchmark = True


def select_highly_variable_genes(adata, n_top_genes=3000):
    n_top_genes = min(int(n_top_genes), int(adata.n_vars))
    try:
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=n_top_genes)
        return
    except Exception as exc:
        print(
            "seurat_v3 highly_variable_genes failed; "
            f"falling back to cell_ranger. error={exc}"
        )

    try:
        sc.pp.highly_variable_genes(adata, flavor="cell_ranger", n_top_genes=n_top_genes)
        if "highly_variable" in adata.var and adata.var["highly_variable"].sum() > 0:
            return
    except Exception as exc:
        print(
            "cell_ranger highly_variable_genes failed; "
            f"falling back to variance ranking. error={exc}"
        )

    x = adata.X
    if scipy_sparse.issparse(x):
        mean = np.asarray(x.mean(axis=0)).ravel()
        mean_sq = np.asarray(x.power(2).mean(axis=0)).ravel()
        gene_var = mean_sq - np.square(mean)
    else:
        gene_var = np.asarray(x).var(axis=0)

    gene_var = np.nan_to_num(gene_var, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
    top_idx = np.argsort(gene_var)[-n_top_genes:]
    highly_variable = np.zeros(adata.n_vars, dtype=bool)
    highly_variable[top_idx] = True
    adata.var["highly_variable"] = highly_variable


class TupleDataset(Dataset):
    '''
    tuple(subgraph, img_patch)
    '''
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args

        self.gene_preprocess() 
        self.image_preprocess() 
        self.loc2graph()

        self.adj_list = build_adjacency_list(
            self.graph.edge_index,
            self.graph.num_nodes,
        )
        if self.args.dataset == "WLP64_65_67":
            return
        elif self.args.dataset == 'DLPFC_batch73':
            return
        else:
            self.construct_interaction()
            return
    

    
    def image_preprocess(self):

        print("Preprocessing images...")
        if self.args.dataset == "Her2st":
            img_dir = os.path.join(self.args.dataset_dir, 'ST-imgs', self.args.slice_name[0], self.args.slice_name)
            img_name = os.listdir(img_dir)[0]
            img_path = os.path.join(img_dir, img_name)
            full_image = cv2.imread(img_path)
        else:
            full_image = cv2.imread(self.args.img_dir)
        full_image = cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB)

        full_image = cv2.GaussianBlur(full_image, (5, 5), 0)

        full_image = cv2.fastNlMeansDenoisingColored(full_image, None, 10, 10, 7, 21)
    
        self.image = full_image
        
        if not self.args.is_10x:
            self.add_image_to_adata(full_image, "data")


        full_image_tensor = torch.tensor(full_image)

        # patch
        patches = []
        assert 'spatial' in self.adata.obsm.keys(), "The dataset does not have spatial coordinates"


        for x, y in self.adata.obsm['spatial']:
            patch = full_image_tensor[y-self.args.patch_size:y+self.args.patch_size, 
                                      x-self.args.patch_size:x+self.args.patch_size]

            patches.append(patch)
        self.patches = patches

        print("Preprocessing images done.")

        return
    
    def add_image_to_adata(self, image, key):
        self.adata.uns["spatial"] = {}
        self.adata.uns['spatial'][key] = {}
        self.adata.uns['spatial'][key]['images'] = {}
        self.adata.uns['spatial'][key]['images']['full_image'] = image
        return



    def gene_preprocess(self):

        print("Preprocessing gene expression matrix...")
        if self.args.dataset == "DLPFC":
            adata = sc.read_visium(path=self.args.dataset_dir, count_file=self.args.countfile_name)
            Ann_df = pd.read_csv(os.path.join(self.args.dataset_dir, '%s_truth.txt'%self.args.countfile_name.split('_')[0]), sep='\t', header=None, index_col=0)
            Ann_df.columns = ['label']
            adata.obs['label'] = Ann_df.loc[adata.obs_names, 'label']
        
        elif self.args.dataset == "mouse_coronal":
            adata = sc.read_visium(path=self.args.dataset_dir, count_file=self.args.countfile_name)
            self.n_clusters = 26

        elif self.args.dataset == "Mouse_Brain_Posterior":
            adata = sc.read_visium(path=self.args.dataset_dir, count_file=self.args.countfile_name)
            self.n_clusters = 26

        elif self.args.dataset == "breast_cancer":
            adata = sc.read_visium(path=self.args.dataset_dir, count_file=self.args.countfile_name)
            self.n_clusters = 20
            Ann_df = pd.read_csv(os.path.join(self.args.dataset_dir, "truth.txt"), sep='\t', header=0, index_col=0)
            Ann_df.columns = ['label']
            missing_indices = adata.obs_names[~adata.obs_names.isin(Ann_df.index)]
           
            adata.obs['label'] = Ann_df['label'].reindex(adata.obs_names, fill_value='Unknown')

        elif self.args.dataset == 'Her2st':
            adata = load_her2st_data(self.args)

        elif self.args.dataset == "MOCHA":
            adata = anndata.read_h5ad(self.args.dataset_dir)
            self.n_clusters = int(self.args.pseudo_cluster)

        elif self.args.dataset == "IDC":
            adata = sc.read_visium(path=self.args.dataset_dir, count_file=self.args.countfile_name)
            self.n_clusters = 5

        elif self.args.dataset == "zebrafishA":
            adata = sc.read_visium(path=self.args.dataset_dir, count_file=self.args.countfile_name)
            self.n_clusters = 13

        elif self.args.dataset == "zebrafishB":
            adata = sc.read_visium(path=self.args.dataset_dir, count_file=self.args.countfile_name)
            self.n_clusters = 20

        elif self.args.dataset == "WS_PLA":
            adata = sc.read_visium(path=self.args.dataset_dir, count_file=self.args.countfile_name)
            self.n_clusters = 10

        elif self.args.dataset == "Xenium":
            adata = anndata.read_h5ad('./dataset/ST+H&E/10x_Xenium_Breast_cancer/cut/1_adata.h5ad')
            self.n_clusters = 14
        

        elif self.args.dataset == "Mouse_Brain_integration":
            adata = anndata.read_h5ad('./dataset/ST+H&E/Mouse_Brain_combine/mouse_anterior_posterior_brain_merged.h5ad')
            self.n_clusters = 26

        elif self.args.dataset == "DLPFC_batch73":
            adata = anndata.read_h5ad('./dataset/DLPFC_7375/3_merged_adata.h5ad')
            adata_adj = anndata.read_h5ad('./dataset/DLPFC_7375/73_75merged_adata_adj.h5ad')
            adata.obsm['adj'] = adata_adj.obsm['adj']
            self.n_clusters = 7

        elif self.args.dataset == "WLP64_65_67":
            adata = anndata.read_h5ad('./dataset/WLP64_65_67/3_merged_adata.h5ad')
            adata_adj = anndata.read_h5ad('./dataset/WLP64_65_67/3_merged_adata_adj.h5ad')
            adata.obsm['adj'] = adata_adj.obsm['adj']
            self.n_clusters = 10


        if "label" in adata.obs.keys():
            self.args.label_available = True
            self.n_clusters = adata.obs.dropna(inplace=False)['label'].nunique()
        adata.var_names_make_unique()

        self.n_spots = adata.shape[0]

        ##normalization
        select_highly_variable_genes(adata, n_top_genes=3000)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)


        self.adata = adata
        #筛选gene后的adata
        self.adata_Vars = self.adata[:, self.adata.var['highly_variable']]

        print("Preprocessing gene expression matrix done.")
        return
    

    def construct_interaction(self):
        """Constructing spot-to-spot interactive graph"""
        position = self.adata.obsm['spatial']
        features = self.adata.X  
    
        distance_matrix = ot.dist(position, position, metric='euclidean')
        n_spot = distance_matrix.shape[0]
    
        self.adata.obsm['distance_matrix'] = distance_matrix
    
        # find k-nearest neighbors
        interaction = np.zeros([n_spot, n_spot])  
        n_neighbors = 6  
    
        for i in range(n_spot):
            vec = distance_matrix[i, :]
            distance_indices = vec.argsort()  
        
            closest_neighbors = []
            for t in range(1, n_neighbors):
                y = distance_indices[t]
                closest_neighbors.append(y)

            best_neighbor = closest_neighbors[0] 
            max_similarity = -1  
        
            for neighbor in closest_neighbors:

                if self.args.dataset == "Her2st":
                    features_i = features[i, :]
                    features_neighbor = features[neighbor, :]  
                
                else:
                    features_i = features[i, :].toarray().ravel()  
                    features_neighbor = features[neighbor, :].toarray().ravel()  


                similarity = 1 - cosine(features_i, features_neighbor) 
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_neighbor = neighbor
        
            interaction[i, best_neighbor] = 1  

        self.adata.obsm['graph_neigh'] = interaction
    
        # transform adj to symmetrical adj
        adj = interaction
        adj = adj + adj.T
        adj = np.where(adj > 1, 1, adj)
    
        self.adata.obsm['adj'] = adj
        
        return 


    def loc2graph(self):
        model, rad_cutoff, k_cutoff = self.args.graph_type, self.args.rad_cutoff, self.args.k_cutoff
        
        assert model in ['Radius', 'KNN'], "You should define how to build the graph"

        print('Building spatial graph...')

        coor = pd.DataFrame(self.adata.obsm['spatial'])  
        coor.index = self.adata.obs.index  
        coor.columns = ['imagerow', 'imagecol']  

        if model == 'Radius':
            assert rad_cutoff is not None, "You should define the radius when building the graph using radius"
            nbrs = sklearn.neighbors.NearestNeighbors(radius=rad_cutoff).fit(coor)
            distances, indices = nbrs.radius_neighbors(coor, return_distance=True) 
            graph_list = []  
            for it in range(indices.shape[0]):
                graph_list.append(pd.DataFrame(zip([it] * indices[it].shape[0], indices[it], distances[it])))

        if model == 'KNN':
            assert k_cutoff is not None, "You should define k when building the graph using KNN"
            nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=k_cutoff + 1).fit(coor)
            distances, indices = nbrs.kneighbors(coor)
            graph_list = []
            for it in range(indices.shape[0]):
                graph_list.append(pd.DataFrame(zip([it] * indices.shape[1], indices[it, :], distances[it, :])))

        graph_df = pd.concat(graph_list) 
        graph_df.columns = ['spot1', 'spot2', 'Distance']

        Spatial_Net = graph_df.copy()
        Spatial_Net = Spatial_Net.loc[Spatial_Net['Distance'] > 0,]
        id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), )) 
        Spatial_Net['spot1'] = Spatial_Net['spot1'].map(id_cell_trans)
        Spatial_Net['spot2'] = Spatial_Net['spot2'].map(id_cell_trans)

        self.adata.uns['Spatial_Net'] = Spatial_Net      
        self.spatial_adj = torch.tensor(graph_df.iloc[:, 0:2].values.T) #spatial graph


        x = self.adata_Vars.X
        if scipy_sparse.issparse(x):
            x = x.toarray()
        elif hasattr(x, "to_numpy"):
            x = x.to_numpy()
        else:
            x = np.asarray(x)
        x = torch.as_tensor(x, dtype=torch.float32)
        self.graph = Data(x=x, edge_index=self.spatial_adj.long())
        self.graph.n_id = torch.arange(self.graph.num_nodes)

        edge_index = self.spatial_adj
        edge_index_np = edge_index.cpu().numpy()  

        self.adata.uns['edge_index'] = edge_index_np

        print('Building graph done.')
        return 
    

    def __getitem__(self, index):
        subgraph = index
        img = self.patches[index]
        return (subgraph, img, index)
    
    def __len__(self):
        return self.n_spots
    
def build_adjacency_list(edge_index, n_nodes):
    edge_index = edge_index.detach().cpu()
    adj = [set() for _ in range(int(n_nodes))]
    for src, dst in edge_index.t().tolist():
        adj[int(src)].add(int(dst))
        adj[int(dst)].add(int(src))
    return [sorted(x) for x in adj]


def graphsage_sample(index, graph, adj_list, num_neighbors):
    seeds = [int(i) for i in index]
    ordered_nodes = list(seeds)
    seen = set(ordered_nodes)
    frontier = list(seeds)

    for layer_k in num_neighbors:
        next_frontier = []
        for node in frontier:
            neighbors = adj_list[int(node)]
            if layer_k is not None and int(layer_k) >= 0:
                neighbors = neighbors[: int(layer_k)]
            for neighbor in neighbors:
                if neighbor not in seen:
                    seen.add(neighbor)
                    ordered_nodes.append(neighbor)
                    next_frontier.append(neighbor)
        frontier = next_frontier
        if not frontier:
            break

    n_id = torch.tensor(ordered_nodes, dtype=torch.long)
    edge_index, _ = pyg_subgraph(
        n_id,
        graph.edge_index,
        relabel_nodes=True,
        num_nodes=graph.num_nodes,
    )
    data = Data(x=graph.x[n_id], edge_index=edge_index)
    data.n_id = n_id
    data.input_id = torch.arange(len(seeds), dtype=torch.long)
    data.batch = torch.arange(len(seeds), dtype=torch.long)
    data.batch_size = len(seeds)
    return data

def collate_fn(batch, dataset):
    assert isinstance(dataset, TupleDataset), "the input dataset should be TupleDataset"
    subgraphs = list(map(lambda x: x[0], batch)) 
    imgs = list(map(lambda x: x[1], batch))
    inds = list(map(lambda x: x[2], batch))

    subgraphs_batch = graphsage_sample(
        subgraphs,
        dataset.graph,
        dataset.adj_list,
        dataset.args.subgraph_neighbors,
    ) #graphsage batch
    imgs_batch = torch.stack(imgs) 
    imgs_batch = imgs_batch.to(torch.float32)
    # [batch_size, 3, patch_size, patch_size]
    imgs_batch = imgs_batch.permute(0, 3, 1, 2)

    return subgraphs_batch, imgs_batch, inds


def load_her2st_data(args):
    path = args.dataset_dir
    name = args.slice_name

    #count
    cnt_path = os.path.join(path, 'ST-cnts', f'{name}.tsv')
    df_cnt = pd.read_csv(cnt_path, sep='\t', index_col=0)
    # location
    pos_path = os.path.join(path, 'ST-spotfiles', f'{name}_selection.tsv')
    df_pos = pd.read_csv(pos_path, sep='\t')
    # label
    lbl_path = os.path.join(path, 'ST-pat/lbl', f'{name}_labeled_coordinates.tsv')
    df_lbl = pd.read_csv(lbl_path, sep='\t')

    df_lbl = df_lbl.dropna(axis=0, how='any')
    df_lbl.loc[df_lbl['label'] == 'undetermined', 'label'] = np.nan
    df_lbl['x'] = (df_lbl['x']+0.5).astype(np.int64)
    df_lbl['y'] = (df_lbl['y']+0.5).astype(np.int64)

    x = df_pos['x'].values
    y = df_pos['y'].values
    ids = []
    for i in range(len(x)):
        ids.append(str(x[i])+'x'+str(y[i])) 
    df_pos['id'] = ids

    x = df_lbl['x'].values
    y = df_lbl['y'].values
    ids = []
    for i in range(len(x)):
        ids.append(str(x[i])+'x'+str(y[i])) 
    df_lbl['id'] = ids

    meta_pos = df_cnt.join(df_pos.set_index('id'))
    meta_lbl = df_cnt.join(df_lbl.set_index('id'))

    adata = anndata.AnnData(df_cnt, dtype=np.int64)
    adata.obsm['spatial'] = np.floor(meta_pos[['pixel_x','pixel_y']].values).astype(int)
    adata.obs['label'] = pd.Categorical(meta_lbl['label'])
    return adata


def plot_spa_cluter_result(adata, title='', color="predict", save_root=None, is_10x=False, size=4):

    if not is_10x:
        sc.pl.spatial(adata, color=color, img_key=None, spot_size=112, scale_factor=1.0)
        ####------H&E image
        #sc.pl.spatial(adata, color=color, img_key="full_image", spot_size=112, scale_factor=1.0)
    else:
        sc.pl.spatial(adata, color=color, img_key="hires", size=size)
    plt.title(title)
    plt.axis('off')
    if save_root:
        plt.savefig(save_root)
    plt.show()
    return

def calculate_entropy(img):
    hist = torch.histc(img, bins=256, min=0, max=255)  

    hist = hist / hist.sum()  
    hist = hist[hist > 0] 

    entropy = -torch.sum(hist * torch.log2(hist))
    return entropy


class ArgumentParserTupleST(argparse.ArgumentParser):
    def set_params(self):

        self.add_argument('--dataset', type=str, default='DLPFC_batch73', help="Dataset")
        self.add_argument('--dataset_dir', type=str, default='', help="ST dataset")
        self.add_argument('--img_dir', type=str, default='./dataset/73-75/tissue_combined_3_images.tif', help='image')
        self.add_argument("--patch_size", type=int, default=56, help="image patch size")
        self.add_argument("--label_available", type=bool, default=False)
        self.add_argument("--is_10x", type=str, default="False")

        # 10x visium
        self.add_argument('--countfile_name', type=str, default='')
        #Her2ST
        self.add_argument('--slice_name', type=str, default='', help="Her2ST")

        ##gene
        self.add_argument("--graph_type", type=str, default="KNN", help="KNN/Radius")
        self.add_argument("--rad_cutoff", type=int, default=None, help="radius")
        self.add_argument("--k_cutoff", type=int, default=6, help="k of knn")
        self.add_argument("--subgraph_neighbors", type=str, default="[6,6]")
        self.add_argument("--gene_in_dim", type=int, default=3000)
        self.add_argument("--gene_hidden_dim", type=int, default=512)
        self.add_argument("--gene_out_dim", type=int, default=128)
        self.add_argument("--gene_num_layers", type=int, default=2)
        self.add_argument("--gene_augtype", type=str, default='noise')
        self.add_argument("--gene_mask_pct", type=float, default=0.1)
        self.add_argument("--gene_sigma_noise", type=int, default=1)
        self.add_argument("--img_sigma_noise", type=int, default=50)
        self.add_argument("--neg_type", type=str, default="others_label")

        self.add_argument("--img_out_dim", type=int, default=128)
        self.add_argument("--fc_dim", type=str, default="[128]")
        self.add_argument("--hidden_dim", type=int, default=128)
        self.add_argument("--mlp_layers", type=int, default=2)
        #pruning
        self.add_argument("--p", type=float, default=1.0)
        self.add_argument("--p_round", type=int, default=1)
        self.add_argument("--iter_epochs", type=int, default=50)

        #train
        self.add_argument("--tau", type=float, default=0.07)
        self.add_argument("--batch_size", type=int, default=60, help="training batch size")
        self.add_argument("--num_epochs", type=int, default=60, help="training epochs")
        self.add_argument("--lr", type=float, default=0.01, help="learning rate")
        self.add_argument("--device", type=str, default="cuda:0", help="training device")
        self.add_argument("--alpha", type=float, default=1.0, help="temperature hyperparameter")

        self.add_argument("--clone", type=int, default=3, help="clone")
        self.add_argument("--internal", type=int, default=3, help="epoch in each clone")
        self.add_argument("--alpha_lr", type=int, default=0.25, help="learning rate of alpha")

        self.add_argument("--reawrd_alphalr", type=int, default=1)
        self.add_argument("--reawrd_emdklr", type=int, default=1)

        self.add_argument("--emb_weight", type=float, default=1.0)
        self.add_argument("--k", type=float, default=6)

        self.add_argument("--valid_cluster", type=int, default=7)
        self.add_argument("--gene_only", type=str, default=False)
        self.add_argument("--img_only", type=str, default=False)


        self.add_argument("--pseudo_cluster", type=int, default=7)

        self.add_argument("--save_dir", type=str, default='./output/DLPFC/')
        self.add_argument("--img_feat", type=str, default=None)
        return

    
def parse_special_params(args):
    args.subgraph_neighbors = eval("".join(args.subgraph_neighbors))
    args.fc_dim = eval("".join(args.fc_dim))
    args.is_first = True 
    args.is_10x = eval(args.is_10x)
    return args

def get_args():
    parser = ArgumentParserTupleST()
    parser.set_params()
    args = parser.parse_args()
    args = parse_special_params(args)
    return args

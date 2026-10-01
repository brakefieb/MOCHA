import numpy as np
import copy, random
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.optim import Adam
import torch.nn.functional as F
from torchvision import transforms
from torch_geometric.nn.models import GraphSAGE
from torch_geometric.transforms import ToSparseTensor
from torch_geometric.utils import subgraph, remove_self_loops
from torch_geometric.nn import GCNConv
import torchvision.transforms.functional as TF
import scanpy as sc
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from utils import collate_fn, calculate_entropy

import warnings
warnings.filterwarnings("ignore")


class GenePipeline(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.gene_enc = GraphSAGE(in_channels=args.gene_in_dim, hidden_channels=args.gene_hidden_dim, 
                                    out_channels=args.gene_out_dim, num_layers=args.gene_num_layers,
                                    norm=nn.BatchNorm1d(args.gene_hidden_dim))
    
    def forward(self, subgraphs):
        '''shape=(n_spot, dim_hidden)'''
        out = self.gene_enc(subgraphs.x, subgraphs.adj_t)

        return out[:subgraphs.batch_size]

class convmixer_block(nn.Module):
    def __init__(self,dim=32,kernel_size=5):
        super().__init__()
        self.dw=nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                nn.BatchNorm2d(dim),
                nn.GELU(),
                nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                nn.BatchNorm2d(dim),
                nn.GELU(),
        )
        self.pw=nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(dim),
        )
    def forward(self,x):
        x=self.dw(x)+x
        x=self.pw(x)
        return x


class Hist2ST_convmixer(nn.Module):
    def __init__(self, img_channels=3, in_channels=32, depth=2, kernel_size0=7, kernel_size1=5) -> None:
        super().__init__()
        self.model = nn.Sequential(nn.Conv2d(img_channels,in_channels,kernel_size0,kernel_size0),
                      nn.Sequential(*[convmixer_block(in_channels,kernel_size1) for _ in range(depth)]),
                      nn.Sequential(nn.Conv2d(in_channels,in_channels//8,1,1),nn.Flatten())
                      )
    def forward(self, imgs):
        return self.model(imgs)
        
class ImagePipeline(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.convs = Hist2ST_convmixer()
        self.convs.model.load_state_dict(torch.load("./model/pretrain/Hist2ST_convmixer.pth"))

        for param in self.convs.model.parameters():
            param.requires_grad = False

        self.fc = nn.Sequential()
        self.fc_dim = self.args.fc_dim

        self.fc.append(nn.Linear(1024, self.fc_dim[0]))
        if len(self.fc_dim) > 1:
            self.fc.append(nn.ReLU())


        for i in range(len(self.fc_dim) - 1): 
            self.fc.append(nn.Linear(self.fc_dim[i], self.fc_dim[i+1]))
            
            if i < len(self.fc_dim) - 2:
                self.fc.append(nn.ReLU())
                self.fc.append(nn.BatchNorm1d(self.fc_dim[i+1]))

        self.img_enc = nn.Sequential(self.convs,
                                     self.fc)

    def forward(self, imgs):
        is_zero_tensor = torch.all(torch.eq(imgs, 0))

        if is_zero_tensor:
            features = self.img_enc(imgs)
            return features
        
        else:

            entropies = []
            weights = []
            for img in imgs:
                entropies.append(calculate_entropy(img.flatten()))

            max_entropy = max(entropies) 
            min_entropy = min(entropies)  

            for entropy in entropies:
                weight = max_entropy - entropy 
                weights.append(weight)

            weights = torch.stack(weights).detach().cpu().numpy()
            weight_range = weights.max() - weights.min()
            if weight_range == 0:
                norm_weights = np.ones_like(weights)
            else:
                norm_weights = (weights - weights.min()) / weight_range

            features = self.img_enc(imgs)
            img_weights = torch.tensor(
                norm_weights,
                dtype=torch.float32,
                device=features.device,
            )

            weighted_features = features * img_weights.unsqueeze(1)  

            return weighted_features


class MultiModalEnc(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.device = args.device
        self.genepipe = GenePipeline(args)
        self.imagepipe = ImagePipeline(args)

        self.concat_dim = args.gene_out_dim + args.img_out_dim 
        

        #MLP
        self.mlp = nn.Linear(self.concat_dim, args.hidden_dim, bias=True)
        init.xavier_normal_(self.mlp.weight.data) 
        init.normal_(self.mlp.bias.data) 

        

    def forward(self, subgraphs, imgs):

        k_weight = nn.Parameter(torch.sigmoid(torch.tensor([self.args.emb_weight]))).to(self.device)

        subgraphs = subgraphs.to(self.device) 
        imgs = imgs.to(self.device) 
        #forward
        gene_featmat = self.genepipe(subgraphs) 
        
        
        img_featmat = self.imagepipe(imgs)

        if gene_featmat.shape[0] > 1:
            gene_featmat = nn.BatchNorm1d(gene_featmat.shape[1]).to(self.args.device)(gene_featmat)
        if img_featmat.shape[0] > 1:
            img_featmat = nn.BatchNorm1d(img_featmat.shape[1]).to(self.args.device)(img_featmat)
    


        if self.args.gene_only == True:
            featmat = gene_featmat

        elif self.args.img_only == True:
            featmat = img_featmat

        else:
            weighted_gene_featmat = k_weight * gene_featmat
            weighted_img_featmat = (1.0 - k_weight) * img_featmat

            # concat
            featmat = torch.concat((weighted_gene_featmat, weighted_img_featmat), dim=-1)

            # MLP
            featmat = self.mlp(featmat)

        
        # for layer in self.mlp:
        #     featmat = layer(featmat)

        return featmat
    
class TupleCL(nn.Module):
    def __init__(self, args, model, dataset, Decoder) -> None:
        super().__init__()
        self.args = args
        self.model = model
        self.dataset = dataset
        self.n = dataset.n_spots 
        self.negpool = None
        self.neginds = None 
        self.device = args.device
        self.Decoder = Decoder
        self.optimizer = Adam(list(model.parameters()) + list(Decoder.parameters()), lr=args.lr)
        return

    def build_mask(self, n_mask, n_genes):
        mask = torch.concatenate([torch.ones(n_mask, dtype=bool), 
                               torch.zeros(n_genes - n_mask, dtype=bool)])
        mask = mask[torch.randperm(mask.shape[0])]
        return mask
    
    

    def graph_aug_neighbors_cat(self, subgraphs):

        adj = self.dataset.adata.obsm['adj']
        adj = torch.tensor(adj).to(self.device)  

        local_adj = adj[subgraphs.n_id][:, subgraphs.n_id]
        local_adj_cpu = local_adj.cpu().numpy()

        n_spot = len(subgraphs.x)
        gene_featmat = copy.copy(subgraphs.x)
        gene_featmat = torch.where(gene_featmat == 0, gene_featmat.mean(dim=1, keepdim=True), gene_featmat)
        gene_featmat_enhanced = torch.clone(subgraphs.x)

        for i in range(n_spot):
            
            neighbors = np.where(local_adj_cpu[i, :] > 0)[0]  

            if len(neighbors) > 10:
                neighbors = random.sample(neighbors, 6)

            if len(neighbors) == 0:
                aug_subgraphs = copy.copy(subgraphs)
                aug_subgraphs.x = gene_featmat_enhanced[i].unsqueeze(0)
                continue

            neighbor_features = subgraphs.x[neighbors]

            neighbor_avg = neighbor_features.mean(dim=0).to(self.device)

            spot_features = subgraphs.x[i].unsqueeze(0).to(self.device)
            gene_featmat_enhanced[i] = (spot_features + neighbor_avg) / 2

        aug_subgraphs = copy.copy(subgraphs)
        aug_subgraphs.x = gene_featmat_enhanced

        '''gene-aug'''
        gene_featmat = gene_featmat_enhanced
        n_genes = gene_featmat.shape[1]

        augtype, mask_pct, sigma_noise = self.args.gene_augtype, self.args.gene_mask_pct, self.args.gene_sigma_noise

        assert mask_pct is not None, "You should specify how many features to mask"
        assert sigma_noise is not None, "You should specify the strongth of noise"
        n_mask = int(n_genes * mask_pct)
        mask = self.build_mask(n_mask, n_genes).to(self.device)
        noise = torch.normal(torch.zeros(n_mask), torch.ones(n_mask) * sigma_noise).to(self.device) 
        gene_featmat[:,mask] += noise  


        aug_subgraphs = copy.copy(subgraphs)
        aug_subgraphs.x = gene_featmat

        return aug_subgraphs

    

    def img_aug(self, imgs):
        '''img-aug'''
        self.img_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomVerticalFlip(p=0.3),
            # transforms.RandomGrayscale(p=0.2),  
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0))], p=0.3),  
            transforms.RandomRotation(degrees=(0, 360))  
        ])

        imgs_aug = self.img_transform(imgs)
        return imgs_aug
    
    def build_neg(self, subgraphs, imgs, inds):
        '''neg'''
        subgraphs_n1, subgraphs_n2 = copy.copy(subgraphs), copy.copy(subgraphs)
        imgs_n1, imgs_n2 = imgs.clone(), imgs.clone()

        if self.args.neg_type == 'origin':

            subgraphs_n1.x = subgraphs_n1.x[torch.randperm(subgraphs_n1.x.shape[0])] 
            imgs_n2 = imgs_n2[torch.randperm(imgs_n2.shape[0])] 

            return [subgraphs_n1, imgs_n1, subgraphs_n2, imgs_n2]
    

        #####-------psudo-label
        elif self.args.neg_type == "others_label":

            spa = self.dataset.adata.obsm['spatial']
            pseudo_imglabels = self.dataset.adata.obs['img_label']

            if self.args.is_first:
                self.negpool = list(range(self.n))
                print("Initializing negpool...")
                for ind in inds:
                    self.negpool.remove(ind)
                random.shuffle(self.negpool)
                self.neginds = self.negpool[:imgs.shape[0]]
                self.args.is_first = False  

            curr_bs = len(inds)  
            assert self.negpool is not None, "the negative sampling pool is None"
            assert self.neginds is not None, "the negative sampling index is None"
            assert len(self.negpool) >= curr_bs, "the size of negative sampling pool should be larger than the current batch size"
    
            neginds = []

            for ind in inds:
                positive_label = pseudo_imglabels[ind]

                available_neg_indices = [
                    idx
                    for idx in self.negpool
                    if pseudo_imglabels[idx] != positive_label and idx not in neginds
                ]
    
                if available_neg_indices:
                    selected_neg_index = random.choice(available_neg_indices)
                    neginds.append(selected_neg_index)

            if len(neginds) < curr_bs:
                self.negpool = list(range(self.n))
                for ind in inds:
                    self.negpool.remove(ind)
                random.shuffle(self.negpool)

                neginds = []

                for ind in inds:
                    positive_label = pseudo_imglabels[ind]
                    available_neg_indices = [
                        idx
                        for idx in self.negpool
                        if pseudo_imglabels[idx] != positive_label and idx not in neginds
                    ]
    
                    if available_neg_indices:
                        selected_neg_index = random.choice(available_neg_indices)
                        neginds.append(selected_neg_index)

            assert len(neginds) >= curr_bs, "The number of negative samples should be at least curr_bs"
            self.neginds = neginds[:curr_bs] 

            neg_data = [self.dataset[negind] for negind in self.neginds]
            subgraphs_n1, imgs_n2, _ = collate_fn(neg_data, self.dataset)
            subgraphs_n1, imgs_n2 = subgraphs_n1.to(self.device), imgs_n2.to(self.device)

            if len(inds) == self.args.batch_size:
                self.negpool = inds

            return [subgraphs_n1, imgs_n1, subgraphs_n2, imgs_n2, subgraphs_n1, imgs_n2]
        
        elif self.args.neg_type == "others":
            spa = self.dataset.adata.obsm['spatial']

            if self.args.is_first:
                self.negpool = list(range(self.n))
                print("Initializing negpool...")
                for ind in inds:
                    self.negpool.remove(ind)
                random.shuffle(self.negpool)
                self.neginds = self.negpool[:imgs.shape[0]]
                self.args.is_first = False  

            curr_bs = len(inds) 
            
            assert self.negpool is not None, "the negative sampling pool is None"
            assert self.neginds is not None, "the negative sampling index is None"
            assert len(self.negpool) >= curr_bs, "the size of negative sampling pool should be larger than the current batch size"
    
            self.neginds = self.negpool[:curr_bs] 

            neg_data = [self.dataset[negind] for negind in self.neginds]
            subgraphs_n1, imgs_n2, _ = collate_fn(neg_data, self.dataset)
            subgraphs_n1, imgs_n2 = subgraphs_n1.to(self.device), imgs_n2.to(self.device)

            if len(inds) == self.args.batch_size:
                self.negpool = inds

            return [subgraphs_n1, imgs_n1, subgraphs_n2, imgs_n2, subgraphs_n1, imgs_n2] 

        else:
            raise KeyError

    def l2_norm(self, feat):
        return F.normalize(feat, p=2, dim=-1)
    
    def forward(self, subgraphs, imgs, inds):
        
        subgraphs, imgs = subgraphs.to(self.device), imgs.to(self.device)
        subgraphs_p = self.graph_aug_neighbors_cat(subgraphs)
        imgs_p = self.img_aug(imgs)

        feat_q = self.model(subgraphs, imgs) 
        feat_p = self.model(subgraphs_p, imgs_p) 

        negs = self.build_neg(subgraphs, imgs, inds)
        
        assert len(negs) % 2 == 0, "the length of negative samples embedding list is odd, it should be even"

        feat_negs = []
        for i in range(len(negs) // 2):
            subgraphs_n, imgs_n = negs[2*i:2*(i+1)]
            subgraphs_n = ToSparseTensor()(subgraphs_n)
            feat_neg = self.model(subgraphs_n, imgs_n)
            feat_negs.append(feat_neg)  

        feat_q, feat_p = self.l2_norm(feat_q), self.l2_norm(feat_p) 
        feat_negs = list(map(lambda x: self.l2_norm(x), feat_negs)) 

        return feat_q, feat_p, feat_negs
    
    def losspart(self, feat_q, feat_p, feat_negs, inds, dataset, alpha):
        device = self.device
        batch_size = feat_q.shape[0]
        
        neg_weights = F.softmax(alpha, dim=0)

        weighted_feat_negs = []
        for i, feat_neg in enumerate(feat_negs):
            weight = neg_weights[i]  
            weighted_feat_negs.append(feat_neg * weight) 
        feat_negs_weighted = torch.cat(weighted_feat_negs, dim=1)  


        #loss_TupleInfoNCE
        cat = torch.cat((feat_p, feat_negs_weighted), 1).view((batch_size, len(neg_weights)+1, -1)) #shape:[batch_size, n_negs+1(正+负样本数量), hidden_dim]
        assert feat_q.dim() == 2
        feat_q = feat_q.unsqueeze(1)
        logits_cl = torch.bmm(feat_q, cat.permute(0,2,1)).squeeze(1)
        labels = torch.zeros((batch_size)).long().to(device)
        loss_cl = torch.nn.CrossEntropyLoss(reduction="mean")(logits_cl/self.args.tau, labels) 

        sub_edgeindex = subgraph(inds, dataset.graph.edge_index)[0] 
        sub_edgeindex = remove_self_loops(sub_edgeindex)[0] 
        src, tar = sub_edgeindex[0, :].numpy(), sub_edgeindex[1,:].numpy()


        inds_map = {inds[i]:i for i in range(len(inds))}
        src_map, tar_map = list(map(lambda x: inds_map[x], src)), list(map(lambda x: inds_map[x], tar)) 

        feat_q = feat_q.squeeze(1)
        logits_recon = F.sigmoid((feat_q[src_map,:] * feat_q[tar_map,:]).sum(axis=1))
        loss_recon = F.mse_loss(logits_recon, torch.ones_like(logits_recon))

        if torch.isnan(loss_recon):
            loss_recon = torch.tensor(0.0, device=loss_recon.device)  
        if torch.isnan(loss_cl):
            loss_cl = torch.tensor(0.0, device=loss_cl.device)

        return loss_cl, loss_recon
    

    def optimize(self, loss_cl, loss_recon):
        self.optimizer.zero_grad()

        loss = loss_cl + (self.args.alpha * loss_recon)

        loss.backward()
                
        self.optimizer.step()

        return
    
    @torch.no_grad()

    def infer_emb(self, dataloader):
        dataset = dataloader.dataset
        n_spot = dataset.n_spots

        #emb_map
        emb = np.zeros((n_spot, self.args.hidden_dim))
        self.model.eval()
        for subgraphs, imgs, inds in dataloader:
            subgraphs = ToSparseTensor()(subgraphs)
            feat_q, _, _ = self(subgraphs, imgs, inds)
            emb[inds, :] = feat_q.detach().cpu().numpy()
        return emb
    

    def validate(self, dataloader):
        dataset = dataloader.dataset
        n_spot = dataset.n_spots
        adata = dataloader.dataset.adata

        emb_1 = np.zeros((n_spot, self.args.hidden_dim))
        emb_2 = np.zeros((n_spot, self.args.hidden_dim))


        self.model.eval()


        for subgraphs, imgs, inds in dataloader: 
            subgraphs, imgs = subgraphs.to(self.device), imgs.to(self.device)
            subgraphs_p = self.graph_aug_neighbors_cat(subgraphs)
            subgraphs_p = ToSparseTensor()(subgraphs_p)

            imgs_p = self.img_aug(imgs)

            subgraphs_n1, subgraphs_n2 = copy.copy(subgraphs_p), copy.copy(subgraphs_p)
            imgs_n1, imgs_n2 = imgs_p.clone(), imgs_p.clone()

            
            imgs_n1 = torch.zeros(imgs_n1.shape).cuda()  
            feat_q1 = self.model(subgraphs_n1, imgs_n1) 
            emb_1[inds, :] = feat_q1.detach().cpu().numpy()

            subgraphs_n2.x.zero_()
            subgraphs_n2.n_id.zero_()
            subgraphs_n2.input_id.zero_()

            feat_q2 = self.model(subgraphs_n2, imgs_n2) 
            emb_2[inds, :] = feat_q2.detach().cpu().numpy()


        emb_df1 = pd.DataFrame(emb_1)
        
        if emb_df1.isna().all().all():
            print("Warning: All NaN!!!!!!!!!!!!!")
            print(emb_df1)
        else:
            if emb_df1.isna().any().any():  
                emb_df1_filled = emb_df1.apply(lambda row: row.fillna(row.mean()), axis=1)
                adata_1 = sc.AnnData(emb_df1_filled)
            else:
                emb_df1 = emb_df1 
                adata_1 = sc.AnnData(emb_df1)

        sc.pp.neighbors(adata_1, n_neighbors=20, use_rep='X')
        resolution = 1.5
        sc.tl.leiden(adata_1, resolution=resolution)

        current_clusters = len(adata_1.obs['leiden'].unique())
        target_clusters = self.args.pseudo_cluster

        low_res = 0.0  
        high_res = 5.0  
        tolerance = 0.5  
        max_iterations = 100
        i = 0

        while abs(current_clusters - target_clusters) > tolerance and i < max_iterations:
            resolution = (low_res + high_res) / 2
            sc.tl.leiden(adata_1, resolution=resolution)
            current_clusters = len(adata_1.obs['leiden'].unique())
            if current_clusters < target_clusters:
                low_res = resolution
            else:
                high_res = resolution

            i += 1

        stpredict = np.array(adata_1.obs['leiden'])

        adata.obs['stpredict'] = stpredict.astype(int)
        adata.obs['stpredict'] = adata.obs['stpredict'].astype('category')

        emb_df2 = pd.DataFrame(emb_2)

        if emb_df2.isna().all().all():
            print("Warning: All NaN!!!!!!!!!!!!!")
            print(emb_df2)
        else:
            if emb_df2.isna().any().any():  
                emb_df2_filled = emb_df2.apply(lambda row: row.fillna(row.mean()), axis=1)
                adata_2 = sc.AnnData(emb_df2_filled)
            else:
                emb_df2 = emb_df2  
                adata_2 = sc.AnnData(emb_df2)

        sc.pp.neighbors(adata_2, n_neighbors=20, use_rep='X')
        resolution = 1.5
        sc.tl.leiden(adata_2, resolution=resolution)

        current_clusters = len(adata_2.obs['leiden'].unique())
        target_clusters = self.args.pseudo_cluster

        low_res = 0.0  
        high_res = 5.0  
        tolerance = 0.5  
        max_iterations = 50
        i = 0

        while abs(current_clusters - target_clusters) > tolerance and i < max_iterations:
            resolution = (low_res + high_res) / 2
            sc.tl.leiden(adata_2, resolution=resolution)
            current_clusters = len(adata_2.obs['leiden'].unique())
            if current_clusters < target_clusters:
                low_res = resolution
            else:
                high_res = resolution

            i += 1

        imgpredict = np.array(adata_2.obs['leiden'])

        adata.obs['imgpredict'] = imgpredict.astype(int)
        adata.obs['imgpredict'] = adata.obs['imgpredict'].astype('category')

        obs_df = adata.obs.dropna()
        ari = adjusted_rand_score(obs_df['stpredict'], obs_df['imgpredict'])

        return ari 
    

    def generate_pseudo_labels(self, dataloader):

        dataset = dataloader.dataset
        n_spot = dataset.n_spots
        adata = dataloader.dataset.adata

        emb_gene = np.zeros((n_spot, self.args.hidden_dim))
        emb_img = np.zeros((n_spot, self.args.hidden_dim))
        self.model.eval()

        for subgraphs, imgs, inds in dataloader: 

            subgraphs, imgs = subgraphs.to(self.device), imgs.to(self.device)
            subgraphs_p = self.graph_aug_neighbors_cat(subgraphs)
            subgraphs_p = ToSparseTensor()(subgraphs_p)

            imgs_p = self.img_aug(imgs)

            subgraphs_n1, subgraphs_n2 = copy.copy(subgraphs_p), copy.copy(subgraphs_p)
            imgs_n1, imgs_n2 = imgs_p.clone(), imgs_p.clone()

            self.args.gene_only = True
            
            imgs_n1 = torch.zeros(imgs_n1.shape).cuda()  
            feat_q1 = self.model(subgraphs_n1, imgs_n1)
            emb_gene[inds, :] = feat_q1.detach().cpu().numpy()

            self.args.gene_only = False

            subgraphs_n2.x.zero_()
            subgraphs_n2.n_id.zero_()
            subgraphs_n2.input_id.zero_()

            self.args.img_only = True

            feat_q2 = self.model(subgraphs_n2, imgs_n2) 
            emb_img[inds, :] = feat_q2.detach().cpu().numpy()

            self.args.img_only = False
        

        emb_df = pd.DataFrame(emb_gene)

        if emb_df.isna().all().all():
            print("Warning: All NaN!!!!!!!!!!!!!")
            print(emb_df)
        else:
            if emb_df.isna().any().any():  
                emb_df_filled = emb_df.apply(lambda row: row.fillna(row.mean()), axis=1)
                adata1 = sc.AnnData(emb_df_filled)
            else:
                emb_df = emb_df  
                adata1 = sc.AnnData(emb_df)

        sc.pp.neighbors(adata1, n_neighbors=20, use_rep='X')

        resolution = 1.5
        sc.tl.leiden(adata1, resolution=resolution)

        current_clusters = len(adata1.obs['leiden'].unique())
        target_clusters = self.args.pseudo_cluster

        low_res = 0.0  
        high_res = 5.0  
        tolerance = 0.5  
        max_iterations = 50
        i = 0

        while abs(current_clusters - target_clusters) > tolerance and i < max_iterations:
            resolution = (low_res + high_res) / 2
            sc.tl.leiden(adata1, resolution=resolution)
            current_clusters = len(adata1.obs['leiden'].unique())

            if current_clusters < target_clusters:
                low_res = resolution
            else:
                high_res = resolution

            i += 1

        gene_label = np.array(adata1.obs['leiden'])

        adata.obs['gene_label'] = gene_label.astype(int)
        adata.obs['gene_label'] = adata.obs['gene_label'].astype('category')


        emb_df2 = pd.DataFrame(emb_img)
        
        if emb_df2.isna().all().all():
            print("Warning: All NaN!!!!!!!!!!!!!")
            print(emb_df2)
        else:
            if emb_df2.isna().any().any():  
                emb_df2_filled = emb_df2.apply(lambda row: row.fillna(row.mean()), axis=1)
                adata2 = sc.AnnData(emb_df2_filled)
            else:
                emb_df2 = emb_df2  
                adata2 = sc.AnnData(emb_df2)

        sc.pp.neighbors(adata2, n_neighbors=20, use_rep='X')
        resolution = 1.5

        sc.tl.leiden(adata2, resolution=resolution)

        current_clusters = len(adata2.obs['leiden'].unique())
        target_clusters = self.args.pseudo_cluster

        low_res = 0.0  
        high_res = 5.0  
        tolerance = 0.5  
        max_iterations = 50
        i = 0

        while abs(current_clusters - target_clusters) > tolerance and i < max_iterations:
            resolution = (low_res + high_res) / 2
            sc.tl.leiden(adata2, resolution=resolution)
            current_clusters = len(adata2.obs['leiden'].unique())

            if current_clusters < target_clusters:
                low_res = resolution
            else:
                high_res = resolution

            i += 1

        img_label = np.array(adata2.obs['leiden'])
        adata.obs['img_label'] = img_label.astype(int)
        adata.obs['img_label'] = adata.obs['img_label'].astype('category')


    
class GCN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = GCNConv(in_channels, out_channels)
        self.prelu = nn.PReLU()

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv(x, edge_index, edge_weight=edge_weight)
        x = self.prelu(x)
        return x

class GCNDecoder(nn.Module):
    def __init__(self, args):
        super(GCNDecoder, self).__init__()

        self.gene_out_dim = args.gene_out_dim  # 128
        self.gene_hidden_dim = args.gene_hidden_dim  # 512
        self.gene_in_dim = args.gene_in_dim  # 3000

        self.conv3 = GCN(self.gene_out_dim, self.gene_hidden_dim)  # GCN: 128 -> 512
        self.conv4 = GCN(self.gene_hidden_dim, self.gene_in_dim)   # GCN: 512 -> 3000

        # self.conv3.conv.lin.weight.data = self.conv4.conv.lin.weight.transpose(0, 1)

    def forward(self, features, edge_index):
        h3 = self.conv3(features, edge_index)  

        h4 = self.conv4(h3, edge_index)         
        
        return h4

import os, random
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
import scanpy as sc
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from torch_sparse import SparseTensor
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
import torch.nn.functional as F
from scipy import sparse
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch_geometric.transforms import ToSparseTensor
from torch.autograd import Variable
from torch.distributions import MultivariateNormal, Normal
from sklearn.metrics import normalized_mutual_info_score
import time
import resource
import pandas as pd

from models import MultiModalEnc, TupleCL, GCNDecoder
from utils import collate_fn, TupleDataset, plot_spa_cluter_result, get_args



def train(args, dataloader):
    # seed_everything()
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


    cl_loss, recon_loss = [], []
    device = args.device
    model = MultiModalEnc(args).to(args.device)
    Decoder = GCNDecoder(args).to(args.device)  # GCNDecoder 
    tuplecl = TupleCL(args, model, dataloader.dataset, Decoder).to(args.device)
    os.makedirs('./model', exist_ok=True)
    os.makedirs('./Decoder', exist_ok=True)

    adata_raw = dataloader.dataset.adata
    adata = adata_raw[:, adata_raw.var['highly_variable']]
    dataloader.dataset.adata = adata
    print('Size of Input: ', adata.shape)
    aris = []

    clone =args.clone
    internal = args.internal

    alpha = Variable(torch.from_numpy(np.array([1, 1, 1])).float(), requires_grad=False)

    weight = torch.tensor(args.emb_weight, dtype=torch.float32, requires_grad=False)

    for epoch in trange(args.num_epochs):

        tuplecl.generate_pseudo_labels(dataloader)

        reward_list = []

        dist = MultivariateNormal(alpha, torch.eye(3) * 1.0)   
        alpha_list = []

        dist_weight = Normal(weight, 1.0)  
        weight_list = []


        for _ in range(clone):
            sample = dist.sample()
            sample_weight = dist_weight.sample()
            
            alpha_list.append(sample)
            weight_list.append(sample_weight)

        baseline_path  = './model/baseline_weight.pth'
        torch.save(model.state_dict(), baseline_path)
        
        baseline_Decoder  = './Decoder/Decoder_weight.pth'
        torch.save(Decoder.state_dict(), baseline_Decoder)


        if epoch % 2 == 0:  # 偶数 epoch
            alpha_base = tuplecl.validate(dataloader)
            candidate_paths = []

            for i in range(clone):
                model.load_state_dict(torch.load(baseline_path))
                Decoder.load_state_dict(torch.load(baseline_Decoder))

                for p in range(internal):
                    loss_cl_b, loss_recon_b = 0, 0
                    
                    global_features = np.zeros((adata.shape[0], args.hidden_dim), dtype=np.float32)

                    for subgraphs, imgs, inds in dataloader: 
                        subgraphs = ToSparseTensor()(subgraphs)   

                        feat_q, feat_p, feat_negs = tuplecl(subgraphs, imgs, inds) 
                        loss_cl, loss_recon = tuplecl.losspart(feat_q, feat_p, feat_negs, inds, dataset, alpha_list[i]) 

                        loss_cl_b = loss_cl_b + loss_cl
                        loss_recon_b = loss_recon_b + loss_recon

                        feat_q_np = feat_q.detach().cpu().numpy()  
                        for idx, ind in enumerate(inds):
                            global_features[ind] = feat_q_np[idx] 
                        
                    global_features_tensor = torch.tensor(global_features).to(device).float()  

                    edge_index = adata.uns['edge_index']
                    edge_index = torch.tensor(edge_index, dtype=torch.long).to(device)

                    reconstructed_matrix = Decoder(global_features_tensor, edge_index)  
                    if sparse.issparse(adata.X):
                        labels_tensor = torch.tensor(adata.X.toarray()).float().to(args.device)
                    else:
                        labels_tensor = torch.tensor(adata.X).float().to(args.device)  

                    loss_recon_m = torch.nn.functional.mse_loss(reconstructed_matrix, labels_tensor)
                    loss_recon_a = loss_recon_b + loss_recon_m
                    tuplecl.optimize(loss_cl_b, loss_recon_a)

                acc = tuplecl.validate(dataloader)
                reward = acc
                # print('reward is ' + str(reward))
                path = f'./model/epoch_{epoch}_clone_{i}.pth'
                torch.save(model.state_dict(), path)

                path_Decoder = f'./Decoder/epoch_{epoch}_clone_{i}.pth'
                torch.save(Decoder.state_dict(), path_Decoder)
                # print("end training，model save at："+path)
                reward_list.append(reward)
                candidate_paths.append((path, path_Decoder))


            dist = MultivariateNormal(alpha, torch.eye(3) * 1.0)   

            eta = args.reawrd_alphalr  
            mu_alpha = alpha.clone()  
            update_alpha = torch.zeros_like(alpha) 

            for k in range(clone):  

                normalized_reward = (reward_list[k] - alpha_base) / alpha_base
                gradient = (alpha_list[k]-alpha)/1.0

                update_alpha += gradient * normalized_reward

            update_alpha /= clone  
            
            mu_alpha += eta * update_alpha  #  mu_alpha

            alpha = mu_alpha

            max_num = reward_list[0]
            max_index = 0

            for i in range(1, len(reward_list)):
                if reward_list[i] > max_num:
                    max_num = reward_list[i]
                    max_index = i

            path, path_Decoder = candidate_paths[max_index]
            model.load_state_dict(torch.load(path))

            Decoder.load_state_dict(torch.load(path_Decoder))


            # print("last-alpha:"+str(alpha))

        else:
            weight_base = tuplecl.validate(dataloader)
            candidate_paths = []

            for i in range(clone):
                model.load_state_dict(torch.load(baseline_path))
                Decoder.load_state_dict(torch.load(baseline_Decoder))
                
                args.emb_weight = weight_list[i]

                for p in range(internal):
                    loss_cl_b, loss_recon_b = 0, 0

                    global_features = np.zeros((adata.shape[0], args.hidden_dim), dtype=np.float32)

                    for subgraphs, imgs, inds in dataloader:
                        subgraphs = ToSparseTensor()(subgraphs)              
                        feat_q, feat_p, feat_negs = tuplecl(subgraphs, imgs, inds) 
                        loss_cl, loss_recon = tuplecl.losspart(feat_q, feat_p, feat_negs, inds, dataset, alpha) 
                        
                        loss_cl_b = loss_cl_b + loss_cl
                        loss_recon_b = loss_recon_b + loss_recon

                        feat_q_np = feat_q.detach().cpu().numpy()  
                        for idx, ind in enumerate(inds):
                            global_features[ind] = feat_q_np[idx]  

                    global_features_tensor = torch.tensor(global_features).to(device).float()  

                    edge_index = adata.uns['edge_index']
                    edge_index = torch.tensor(edge_index, dtype=torch.long).to(device)

                    reconstructed_matrix = Decoder(global_features_tensor, edge_index)  
                    if sparse.issparse(adata.X):
                        labels_tensor = torch.tensor(adata.X.toarray()).float().to(args.device)
                    else:
                        labels_tensor = torch.tensor(adata.X).float().to(args.device)  
            
                    loss_recon_m = torch.nn.functional.mse_loss(reconstructed_matrix, labels_tensor)

                    loss_recon_a = loss_recon_b + loss_recon_m

                    tuplecl.optimize(loss_cl_b, loss_recon_a)


                acc = tuplecl.validate(dataloader)
                reward = acc
                # print('reward is ' + str(reward))
                path = f'./model/epoch_{epoch}_clone_{i}.pth'
                torch.save(model.state_dict(), path)

                path_Decoder = f'./Decoder/epoch_{epoch}_clone_{i}.pth'
                torch.save(Decoder.state_dict(), path_Decoder)
                # print("end training，model save at："+path)
                reward_list.append(reward)
                candidate_paths.append((path, path_Decoder))
            

            dist_weight = Normal(weight, 1.0)  
            eta = args.reawrd_emdklr  
            mu_weight = weight.clone()  
            update_weight = torch.zeros_like(weight)  

            
            for k in range(clone):  

                normalized_reward = (reward_list[k] - weight_base) / weight_base

                gradient = (weight_list[k]-weight)/1.0

                update_weight += gradient * normalized_reward
               
            update_weight /= clone  
            
            mu_weight += eta * update_weight  

            weight = mu_weight

            max_num = reward_list[0]
            max_index = 0

            for i in range(1, len(reward_list)):
                if reward_list[i] > max_num:
                    max_num = reward_list[i]
                    max_index = i

            path, path_Decoder = candidate_paths[max_index]
            model.load_state_dict(torch.load(path))

            Decoder.load_state_dict(torch.load(path_Decoder))

            # print("last-weight:"+str(weight))

            args.emb_weight = weight
            # print("last-emb_weight:"+str(args.emb_weight))

            w=torch.sigmoid(args.emb_weight).detach().cpu().numpy()

            try:
                existing_w = np.load(os.path.join(args.save_dir, "emd_k.npy"))
            except FileNotFoundError:
                existing_w = np.array([])

            existing_w = np.append(existing_w, [w])

            np.save(os.path.join(args.save_dir, "emd_k.npy"), existing_w)


        loss_cl_b, loss_recon_b = 0, 0

        global_features = np.zeros((adata.shape[0], args.hidden_dim), dtype=np.float32)

        for subgraphs, imgs, inds in dataloader: 

            # contrastive learning
            subgraphs = ToSparseTensor()(subgraphs)  ##transfer data to sparse data which can ensure the reproducibility when seed fixed
            feat_q, feat_p, feat_negs = tuplecl(subgraphs, imgs, inds) 
            
            # print("used-alpha:"+str(alpha))
            # print("used-emd_k:"+str(args.emb_weight))
            loss_cl, loss_recon = tuplecl.losspart(feat_q, feat_p, feat_negs, inds, dataset, alpha) #计算对比损失，优化网络参数


            loss_cl_b = loss_cl_b + loss_cl
            loss_recon_b = loss_recon_b + loss_recon

            feat_q_np = feat_q.detach().cpu().numpy()  
            for idx, ind in enumerate(inds):
                global_features[ind] = feat_q_np[idx]  

            torch.cuda.empty_cache()
                        
        global_features_tensor = torch.tensor(global_features).to(device).float()  

        edge_index = adata.uns['edge_index']
        edge_index = torch.tensor(edge_index, dtype=torch.long).to(device)

        reconstructed_matrix = Decoder(global_features_tensor, edge_index)  
        if sparse.issparse(adata.X):
            labels_tensor = torch.tensor(adata.X.toarray()).float().to(args.device)
        else:
            labels_tensor = torch.tensor(adata.X).float().to(args.device)  

        loss_recon_m = torch.nn.functional.mse_loss(reconstructed_matrix, labels_tensor)
        loss_recon_a = loss_recon_b + loss_recon_m

        tuplecl.optimize(loss_cl_b, loss_recon_a)

        cl_loss_bc = loss_cl_b.cpu().detach().numpy()
        cl_loss.append(cl_loss_bc)
        loss_recon_ac = loss_recon_a.cpu().detach().numpy()
        recon_loss.append(loss_recon_ac)

        

    emb = tuplecl.infer_emb(dataloader)

    adata.obsm['X_emb'] = emb 

    emb_df = pd.DataFrame(emb)
    adata1 = sc.AnnData(emb_df)
    sc.pp.neighbors(adata1, n_neighbors=20, use_rep='X')

    resolution = 1.5

    sc.tl.leiden(adata1, resolution=resolution)

    current_clusters = len(adata1.obs['leiden'].unique())
    target_clusters = args.pseudo_cluster

    low_res = 0.0  
    high_res = 5.0  
    tolerance = 0.5  
    max_iterations = 50
    i = 0

    while abs(current_clusters - target_clusters) > tolerance and i < max_iterations:
        resolution = (low_res + high_res) / 2
        sc.tl.leiden(adata1, resolution=resolution)
        current_clusters = len(adata1.obs['leiden'].unique())
        # print(f"Current resolution: {resolution}, Cluster count: {current_clusters}")

        if current_clusters < target_clusters:
            low_res = resolution
        else:
            high_res = resolution

        i += 1

    predict = np.array(adata1.obs['leiden'])

    adata.obs['predict'] = predict.astype(int)
    adata.obs['predict'] = adata.obs['predict'].astype('category')

    if args.slice_name:
        title = args.slice_name
    else:
        title = args.countfile_name
        
    adata.write(os.path.join(args.save_dir, (title + "_adata.h5ad")))

    reconstructed_matrix_np = reconstructed_matrix.cpu().detach().numpy()  # transfer to CPU and convert


    reconstructed_df = pd.DataFrame(reconstructed_matrix_np)

    output_file_path = os.path.join(args.save_dir, (title + "_reconstructed_matrix.csv"))  
    reconstructed_df.to_csv(output_file_path, index=False)  

    
    return






if __name__ == "__main__":

    start_time = time.time()
    start_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"start time: {start_time} s")

    args = get_args()

    print(args.slice_name)
    print(args)

    dataset = TupleDataset(args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=lambda x: collate_fn(x, dataset), shuffle=True)

    train(args, dataloader)

    end_time = time.time()
    end_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    elapsed_time = end_time - start_time
    memory_usage = end_memory - start_memory

    print(f"start memory: {start_memory} KB")
    print(f"end memory: {end_memory} KB")

    print(f"used time: {elapsed_time} s")
    print(f"used memory: {memory_usage} KB")
   
    
    

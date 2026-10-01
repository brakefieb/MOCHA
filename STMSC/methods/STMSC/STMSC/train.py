from .networks import DeconvNet1,Encoder_map
from .utils import LL_loss
import torch
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np

def train_stmsc_model(adata_st, adata_basis, device='cuda:0', epochs=5000, lr=0.001):
    X = torch.tensor(adata_st.X.toarray()).float().to(device)
    A = torch.tensor(adata_st.obsm["graph"]).float().to(device)
    Y = torch.tensor(adata_st.obsm["count"]).float().to(device)
    lY = torch.tensor(adata_st.obs["library_size"].values.reshape(-1, 1)).float().to(device)
    slice_ids = torch.tensor(adata_st.obs["slice"].values).long().to(device)
    basis = torch.tensor(adata_basis.X).float().to(device)

    model = DeconvNet1(
        hidden_dims=[X.shape[1], 512, 128],
        n_celltypes=adata_basis.shape[0],
        n_slices=len(set(slice_ids.tolist())),
        n_heads=1, slice_emb_dim=16, coef_fe=0.1).to(device)

    optimizer = optim.Adamax(model.parameters(), lr=lr)
    for step in tqdm(range(epochs)):
        recon, Z = model(A, X, Y, lY, slice_ids, basis)
        loss = torch.mean(torch.sqrt(torch.sum((X - recon) ** 2, dim=1)))
        optimizer.zero_grad(); loss.backward(); optimizer.step()

    model.eval()
    Z = model.evaluate(A, X, slice_ids)
    return model, Z.detach().cpu().numpy()


def learn_mapping_matrix(adata_st, adata_basis, lam=7, device='cuda:0', epoch=5000):

    X = torch.from_numpy(adata_st.X.toarray()).float().to(device)
    Y = torch.from_numpy(adata_st.obsm["count"]).float().to(device)
    lY = torch.from_numpy(adata_st.obs["library_size"].values.reshape(-1, 1)).float().to(device)
    slice_ids = torch.from_numpy(adata_st.obs["slice"].values).long().to(device)
    basis = torch.from_numpy(adata_basis.X).float().to(device)

    lamda1 = lam / 10.0
    lamda2 = 1 - lamda1

    emb_sp = F.normalize(X, p=2, eps=1e-12, dim=1)
    emb_sc = F.normalize(basis, p=2, eps=1e-12, dim=1)

    model_map = Encoder_map(basis.shape[0], X.shape[0]).to(device)
    optimizer_map = torch.optim.Adam(model_map.parameters(), lr=0.01)

    for step in tqdm(range(epoch)):
        model_map.train()
        map_matrix = model_map()
        loss_recon, loss_NCE = LL_loss(adata_st, emb_sp, emb_sc, map_matrix, device)
        loss = lamda1 * loss_recon + lamda2 * loss_NCE
        optimizer_map.zero_grad()
        loss.backward()
        optimizer_map.step()

    with torch.no_grad():
        model_map.eval()
        map_matrix_soft = F.softmax(model_map(), dim=1).cpu().numpy()  # shape: cell x spot
        adata_st.obsm['emb_sp'] = emb_sp.cpu().numpy()
        adata_basis.obsm['emb_sc'] = emb_sc.cpu().numpy()
        adata_st.obsm['map_matrix'] = map_matrix_soft.T 

    return map_matrix.cpu().detach().numpy().T 
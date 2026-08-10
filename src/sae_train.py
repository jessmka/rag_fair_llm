import torch
import torch.nn as nn

import numpy as np

# TODO []: Not full functional yet - need to output the SAE features once I've figured out what to do with them

class SAE(nn.Module):
    def __init__(self, d_in, d_latent):
        super().__init__()
        self.enc = nn.Linear(d_in, d_latent, bias=True)
        self.dec = nn.Linear(d_latent, d_in, bias=True)

    def forward(self, x):
        latent = torch.relu(self.enc(x))
        recon = self.dec(latent)
        return recon, latent

embeddings = np.load('data/mind_abstract_embeddings.npy', allow_pickle=True)
d_in = embeddings.shape[1]
d_latent = d_in * 8   # overcomplete factor, tune this
sae = SAE(d_in, d_latent)

opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
X = torch.tensor(embeddings, dtype=torch.float32)
l1_coef = 1e-3

for epoch in range(50):
    perm = torch.randperm(X.shape[0])
    total_loss = 0
    for i in range(0, X.shape[0], 256):
        batch = X[perm[i:i+256]]
        recon, latent = sae(batch)
        recon_loss = ((recon - batch) ** 2).mean()
        sparsity_loss = latent.abs().mean()
        loss = recon_loss + l1_coef * sparsity_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()
    print(f"epoch {epoch}: loss {total_loss:.4f}")

sae.eval()
with torch.no_grad():
    _, all_latents = sae(X)
all_latents = all_latents.numpy()   # shape (n_articles, d_latent)
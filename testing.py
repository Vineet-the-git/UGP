import torch
from scvi.models.proteinvi_unshared import PROTENVI_UNSHARED

if __name__ == "__main__":
    print("Hello from testing.py")
    #create a random tensor of whole numbers of shape (4, 224)
    x = torch.randint(0, 10, (4, 224))
    #create a model
    model = PROTENVI_UNSHARED(x.shape[1], latent_distribution="normal")
    #forward pass
    recon_loss, kl_div_z, kl_div_back_pro, z = model(x)
    print(recon_loss.shape, kl_div_z.shape, kl_div_back_pro.shape, z.shape)

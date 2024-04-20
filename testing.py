import torch as th
from scvi.models.midas import Net

rna_encoder_dims = [1191, 1024]
adt_encoder_dims = [54, 1024]
shared_encoder_dims = [180, 1024]
common_encoder_dims = [1024, 128, 68]
common_decoder_dims = [34, 128, 1024]
s_encoder_dims = [2, 16, 16, 68]
s_decoder_dims = [2, 16, 16, 2]
norm = "ln"
drop = 0.2
dim_z = 34

#write testing code
if __name__ == "__main__":
    #initialize model
    model = Net(rna_encoder_dims, adt_encoder_dims, shared_encoder_dims, common_encoder_dims, common_decoder_dims, s_encoder_dims, s_decoder_dims, norm, drop, dim_z)
    #initialize input
    inputs = {}
    inputs["x_unshared"] = th.randn(256, 1191)
    inputs["x_shared"] = th.randn(256, 180)
    inputs["y_unshared"] = th.randn(256, 54)
    inputs["y_shared"] = th.randn(256, 180)
    print(inputs["x_unshared"].dtype)
    #run model
    loss, z = model(inputs)
    print(loss)
    print(z.shape)

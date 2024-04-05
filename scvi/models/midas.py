from os import path
from os.path import join as pj
import math
import numpy as np
import torch as th
import torch.nn as nn
import scvi.models.utils_midas as utils
import torch.nn.functional as F

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

class Net(nn.Module):
    def __init__(self, rna_encoder_dims, adt_encoder_dims, shared_encoder_dims, common_encoder_dims, common_decoder_dims, s_encoder_dims, s_decoder_dims, norm, drop, dim_z):
        super(Net, self).__init__()
        self.rna_encoder_dims = rna_encoder_dims
        self.adt_encoder_dims = adt_encoder_dims
        self.shared_encoder_dims = shared_encoder_dims
        self.common_encoder_dims = common_encoder_dims
        self.common_decoder_dims = common_decoder_dims
        self.s_encoder_dims = s_encoder_dims
        self.s_decoder_dims = s_decoder_dims
        self.norm = norm
        self.drop = drop
        self.dim_z = dim_z
        self.sct = SCT(rna_encoder_dims, adt_encoder_dims, shared_encoder_dims, common_encoder_dims, common_decoder_dims, s_encoder_dims, s_decoder_dims, norm, drop, dim_z)
        self.loss_calculator = LossCalculator()

    def forward(self, inputs):
        x_r_pre, z_x_mu, z_x_logvar, z = self.sct(inputs)
        loss = self.loss_calculator(inputs, x_r_pre, z_x_mu, z_x_logvar, z)
        return loss, z

class SCT(nn.Module):
    def __init__(self, rna_encoder_dims, adt_encoder_dims, shared_encoder_dims, common_encoder_dims, common_decoder_dims, s_encoder_dims, s_decoder_dims, norm, drop, dim_z):
        super(SCT, self).__init__()
        self.rna_encoder_dims = rna_encoder_dims
        self.adt_encoder_dims = adt_encoder_dims
        self.shared_encoder_dims = shared_encoder_dims
        self.common_encoder_dims = common_encoder_dims
        self.common_decoder_dims = common_decoder_dims
        self.s_encoder = s_encoder_dims
        self.s_decoder = s_decoder_dims
        self.norm = norm
        self.drop = drop
        self.dim_z = dim_z
        self.sampling = False
        self.batch_correction = False
        self.b_centroid = None

        x_encs = {}
        x_common_enc = MLP(common_encoder_dims, hid_norm=norm, hid_drop=drop)
        
        # Individual modality encoders q(z|x^m)
        x_unshared_rna_enc = MLP(rna_encoder_dims, hid_norm=norm, hid_drop=drop)
        x_encs["rna"] = nn.Sequential(x_unshared_rna_enc, x_common_enc)

        x_shared_enc = MLP([shared_encoder_dims[0]+1] + shared_encoder_dims[1:], hid_norm=norm, hid_drop=drop)
        x_encs["shared"] = nn.Sequential(x_shared_enc, x_common_enc)

        x_unshared_adt_enc = MLP(adt_encoder_dims, hid_norm=norm, hid_drop=drop)
        x_encs["adt"] = nn.Sequential(x_unshared_adt_enc, x_common_enc)

        self.x_encs = nn.ModuleDict(x_encs)

        # Modality decoder p(x^m|c, b)
        x_decos = {}
        x_common_dec = MLP(common_decoder_dims, hid_norm=norm, hid_drop=drop)

        # Individual modality decoders p(x^m|z)
        x_unshared_rna_dec = MLP(rna_encoder_dims[::-1], hid_norm=norm, hid_drop=drop)
        x_decos["rna"] = nn.Sequential(x_common_dec, x_unshared_rna_dec)

        x_shared_dec = MLP(shared_encoder_dims[::-1], hid_norm=norm, hid_drop=drop)
        x_decos["shared"] = nn.Sequential(x_common_dec, x_shared_dec)

        x_unshared_adt_dec = MLP(adt_encoder_dims[::-1], hid_norm=norm, hid_drop=drop)
        x_decos["adt"] = nn.Sequential(x_common_dec, x_unshared_adt_dec)

        self.x_decos = nn.ModuleDict(x_decos)

        # Batch encoder q(z|s)
        # self.s_enc = MLP(s_encoder_dims, hid_norm=norm, hid_drop=drop)
        # Batch decoder p(s|b)
        # self.s_dec = MLP(s_decoder_dims, hid_norm=norm, hid_drop=drop)

    def forward(self, inputs):
        x_unshared = inputs["x_unshared"]
        x_shared = inputs["x_shared"]
        y_unshared = inputs["y_unshared"]
        y_shared = inputs["y_shared"]
        x_batch_id = inputs["x_batch_id"]
        y_batch_id = inputs["y_batch_id"]

        # Encode x_m
        z_x_mu, z_x_logvar = {}, {}
        x_pp = {}

        x_shared = th.cat([x_batch_id, x_shared], dim=1)
        y_shared = th.cat([y_batch_id, y_shared], dim=1)
        x_pp["rna"] = preprocess(x_unshared)
        x_pp["shared"] = th.cat([preprocess(x_shared), preprocess(y_shared)], dim=0)
        x_pp["adt"] = preprocess(y_unshared)

        # encoding
        for m in x_pp.keys():
            # if m=="rna":
            #     assert x_pp[m].size() == th.Size([256, 1191]), f"Expected size [256, 1191], got {x_pp[m].size()}"
            # elif m=="shared":
            #     assert x_pp[m].size() == th.Size([512, 180]), f"Expected size [256, 180], got {x_pp[m].size()}"
            # elif m=="adt":
            #     assert x_pp[m].size() == th.Size([256, 54]), f"Expected size [256, 54], got {x_pp[m].size()}"
            h = self.x_encs[m](x_pp[m])
            z_x_mu[m], z_x_logvar[m] = h.split(self.dim_z, dim=1)
        
        # Sample z
        if self.training:
            z_unshared_rna = utils.sample_gaussian(z_x_mu["rna"], z_x_logvar["rna"])
            z_unshared_adt = utils.sample_gaussian(z_x_mu["adt"], z_x_logvar["adt"])
            z_shared = utils.sample_gaussian(z_x_mu["shared"], z_x_logvar["shared"])
        else:  # validation
            z_unshared_rna = z_x_mu["rna"]
            z_unshared_adt = z_x_mu["adt"]
            z_shared = z_x_mu["shared"]

        # Generate x_m activation/probability
        x_r_pre = {}
        x_r_pre["rna"] = self.x_decos["rna"](z_unshared_rna)
        x_r_pre["shared"] = self.x_decos["shared"](z_shared)
        x_r_pre["adt"] = self.x_decos["adt"](z_unshared_adt)

        z = {}
        z["unshared_rna"] = z_unshared_rna
        z["shared"] = z_shared
        z["unshared_adt"] = z_unshared_adt
        return x_r_pre, z_x_mu, z_x_logvar, z

class LossCalculator(nn.Module):

    def __init__(self):
        super(LossCalculator, self).__init__()
        # self.log_softmax = func("log_softmax")
        # self.nll_loss = nn.NLLLoss(reduction='sum')
        self.pois_loss = nn.PoissonNLLLoss(full=True, reduction='none')
        self.bce_loss = nn.BCELoss(reduction='none')
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')  # log_softmax + nll
        self.mse_loss = nn.MSELoss(reduction='none')
        self.kld_loss = nn.KLDivLoss(reduction='sum')
        self.gaussian_loss = nn.GaussianNLLLoss(full=True, reduction='sum')
        # self.enc_s = MLP([o.dim_s]+o.dims_enc_s+[o.dim_b*2], hid_norm=o.norm, hid_drop=o.drop)


    def forward(self, inputs, x_r_pre, z_x_mu, z_x_logvar, z):
        x = {}
        x["rna"] = inputs["x_unshared"]
        x["shared"] = th.cat([inputs["x_shared"], inputs["y_shared"]], dim=0)
        x["adt"] = inputs["y_unshared"]

        loss_recon = self.calc_recon_loss(x, x_r_pre)
        loss_kld_z = self.calc_kld_z_loss(z_x_mu, z_x_logvar)
        clip_loss_rna = self.CLIP_loss(z["unshared_rna"], z["shared"][:z["unshared_rna"].size(0)])
        clip_loss_adt = self.CLIP_loss(z["unshared_adt"], z["shared"][z["unshared_rna"].size(0):])
        
        # if debug == 1:
        #     print("recon: %.3f\tkld_z: %.3f\ttopo: %.3f" % (loss_recon.item(),
        #         loss_kld_z.item(), loss_mod.item()))
        return 0.1 * loss_recon + 0.001*loss_kld_z + clip_loss_rna + clip_loss_adt


    def calc_recon_loss(self, x, x_r_pre):
        losses = {}
        # Reconstruciton losses of x^m
        for m in x.keys():
            losses[m] = (self.pois_loss(x_r_pre[m], x[m])).sum()/x[m].size(0)

        # print(losses)
        return sum(losses.values())/len(losses.keys())


    def calc_kld_z_loss(self, mu, logvar):
        kld_losses = {}
        for m in mu.keys():
            kld_losses[m] = self.calc_kld_loss(mu[m], logvar[m])
        return sum(kld_losses.values())/len(kld_losses.keys())

    def calc_kld_loss(self, mu, logvar):
        return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum() / mu.size(0)


    def calc_mod_align_loss(self, z_uni):
        z_uni_stack = th.stack(list(z_uni.values()), dim=0)  # M * N * K
        z_uni_mean = z_uni_stack.mean(0, keepdim=True)  # 1 * N * K
        return ((z_uni_stack - z_uni_mean)**2).sum() / z_uni_stack.size(1)
    
    def l2_normalize_batch_pytorch(self, batch):
        norm = th.norm(batch, dim=1, keepdim=True)
        normalized_batch = batch / norm
        return normalized_batch
    
    def cross_entropy(self, preds, targets, reduction='none'):
        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()

    def CLIP_loss(self, rna_total,rna_unshared):
        text_embeddings = self.l2_normalize_batch_pytorch(rna_total)
        image_embeddings = self.l2_normalize_batch_pytorch(rna_unshared)
        temperature = 1.0
        logits = (text_embeddings @ image_embeddings.T) 
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2, dim=-1
        )
        texts_loss = self.cross_entropy(logits, targets, reduction='none')
        images_loss = self.cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + texts_loss) / 2.0 
        return loss.mean()


class MLP(nn.Module):
    def __init__(self, features=[], hid_trans='mish', out_trans=False,
                 norm=False, hid_norm=False, drop=False, hid_drop=False):
        super(MLP, self).__init__()
        layer_num = len(features)
        assert layer_num > 1, "MLP should have at least 2 layers!"
        if norm:
            hid_norm = out_norm = norm
        else:
            out_norm = False
        if drop:
            hid_drop = out_drop = drop
        else:
            out_drop = False
        
        layers = []
        for i in range(1, layer_num):
            layers.append(nn.Linear(features[i-1], features[i]))
            if i < layer_num - 1:  # hidden layers (if layer number > 2)
                layers.append(Layer1D(features[i], hid_norm, hid_trans, hid_drop))
            else:                  # output layer
                layers.append(Layer1D(features[i], out_norm, out_trans, out_drop))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    
class Layer1D(nn.Module):
    def __init__(self, dim=False, norm=False, trans=False, drop=False):
        super(Layer1D, self).__init__()
        layers = []
        if norm == "bn":
            layers.append(nn.BatchNorm1d(dim))
        elif norm == "ln":
            layers.append(nn.LayerNorm(dim))
        if trans:
            layers.append(func(trans))
        if drop:
            layers.append(nn.Dropout(drop))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    
def preprocess(x):
    x = x.log1p()
    return x  

def norm_grad(input, max_norm):
    if input.requires_grad:
        def norm_hook(grad):
            N = grad.size(0)  # batch number
            norm = grad.view(N, -1).norm(p=2, dim=1) + 1e-6
            scale = (norm / max_norm).clamp(min=1).view([N]+[1]*(grad.dim()-1))
            return grad / scale

            # clip_coef = float(max_norm) / (grad.norm(2).data[0] + 1e-6)
            # return grad.mul(clip_coef) if clip_coef < 1 else grad
        input.register_hook(norm_hook)


def clip_grad(input, value):
    if input.requires_grad:
        input.register_hook(lambda g: g.clamp(-value, value))


def scale_grad(input, scale):
    if input.requires_grad:
        input.register_hook(lambda g: g * scale)

def exp(x, eps=1e-12):
    return (x < 0) * (x.clamp(max=0)).exp() + (x >= 0) / ((-x.clamp(min=0)).exp() + eps)


def log(x, eps=1e-12):
    return (x + eps).log()


def func(func_name):
    if func_name == 'tanh':
        return nn.Tanh()
    elif func_name == 'relu':
        return nn.ReLU()
    elif func_name == 'silu':
        return nn.SiLU()
    elif func_name == 'mish':
        return nn.Mish()
    elif func_name == 'sigmoid':
        return nn.Sigmoid()
    elif func_name == 'softmax':
        return nn.Softmax(dim=1)
    elif func_name == 'log_softmax':
        return nn.LogSoftmax(dim=1)
    else:
        assert False, "Invalid func_name."


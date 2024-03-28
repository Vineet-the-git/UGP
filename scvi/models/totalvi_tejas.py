# -*- coding: utf-8 -*-
"""Main module."""
from typing import Dict, Optional, Tuple, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Bernoulli, kl_divergence as kl

from torch.autograd import Variable

from scvi.models.distributions import ZeroInflatedNegativeBinomial, NegativeBinomial
from scvi.models.log_likelihood import log_mixture_nb
from scvi.models.modules import DecoderTOTALVI, EncoderTOTALVI,DecoderTOTALVI_shared,FCLayers,DecoderSCVI,DecoderproteinVI,EncoderProteinVI,Encoder,DecoderTOTALVIRNA,DecoderproteinVI_sir,DecoderTOTALVIRNA_sir,DecoderTOTALVI_shared_sir
from scvi.models.utils import one_hot
import numpy as np
import collections
torch.backends.cudnn.benchmark = True

def reparameterize_gaussian(mu, var):
    return Normal(mu, var.sqrt()).rsample()

# VAE model
class TOTALVI(nn.Module):
    """Total variational inference for CITE-seq data

    Implements the totalVI model of [Gayoso19]_.

    :param n_input_genes: Number of input genes
    :param n_input_proteins: Number of input proteins
    :param n_batch: Number of batches
    :param n_labels: Number of labels
    :param n_hidden: Number of nodes per hidden layer for the z encoder (protein+genes),
                     genes library encoder, z->genes+proteins decoder
    :param n_latent: Dimensionality of the latent space
    :param n_layers: Number of hidden layers used for encoder and decoder NNs
    :param dropout_rate: Dropout rate for neural networks
    :param genes_dispersion: One of the following

        * ``'gene'`` - genes_dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - genes_dispersion can differ between different batches
        * ``'gene-label'`` - genes_dispersion can differ between different labels

    :param protein_dispersion: One of the following

        * ``'protein'`` - protein_dispersion parameter is constant per protein across cells
        * ``'protein-batch'`` - protein_dispersion can differ between different batches NOT TESTED
        * ``'protein-label'`` - protein_dispersion can differ between different labels NOT TESTED

    :param log_variational: Log(data+1) prior to encoding for numerical stability. Not normalization.
    :param reconstruction_loss_genes:  One of

        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution

    :param latent_distribution:  One of

        * ``'normal'`` - Isotropic normal
        * ``'ln'`` - Logistic normal with normal params N(0, 1)

    Examples:
        >>> dataset = Dataset10X(dataset_name="pbmc_10k_protein_v3", save_path=save_path)
        >>> totalvae = TOTALVI(gene_dataset.nb_genes, len(dataset.protein_names), use_cuda=True)
    """

    def __init__(
        self,
        n_input_genes: int,
        n_input_proteins: int,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 256,
        n_latent: int = 20,
        n_layers_encoder: int = 1,
        n_layers_decoder: int = 1,
        dropout_rate_decoder: float = 0.2,
        dropout_rate_encoder: float = 0.2,
        gene_dispersion: str = "gene",
        protein_dispersion: str = "protein",
        log_variational: bool = True,
        reconstruction_loss_gene: str = "nb",
        latent_distribution: str = "ln",
        protein_batch_mask: List[np.ndarray] = None,
        encoder_batch: bool = True,
    ):
        super().__init__()
        self.gene_dispersion = gene_dispersion
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.reconstruction_loss_gene = reconstruction_loss_gene
        self.n_batch = n_batch
        self.n_labels = n_labels
        self.n_input_genes = n_input_genes
        self.n_input_proteins = n_input_proteins
        self.protein_dispersion = protein_dispersion
        self.latent_distribution = latent_distribution
        self.protein_batch_mask = protein_batch_mask

        # parameters for prior on rate_back (background protein mean)
        if n_batch > 0:
            self.background_pro_alpha = torch.nn.Parameter(
                torch.randn(n_input_proteins, n_batch)
            )
            self.background_pro_log_beta = torch.nn.Parameter(
                torch.clamp(torch.randn(n_input_proteins, n_batch), -10, 1)
            )
        else:
            self.background_pro_alpha = torch.nn.Parameter(
                torch.randn(n_input_proteins)
            )
            self.background_pro_log_beta = torch.nn.Parameter(
                torch.clamp(torch.randn(n_input_proteins), -10, 1)
            )

        if self.gene_dispersion == "gene":
            self.px_r = torch.nn.Parameter(torch.randn(n_input_genes))
        elif self.gene_dispersion == "gene-batch":
            self.px_r = torch.nn.Parameter(torch.randn(n_input_genes, n_batch))
        elif self.gene_dispersion == "gene-label":
            self.px_r = torch.nn.Parameter(torch.randn(n_input_genes, n_labels))
        else:  # gene-cell
            pass

        if self.protein_dispersion == "protein":
            self.py_r = torch.nn.Parameter(torch.ones(self.n_input_proteins))
        elif self.protein_dispersion == "protein-batch":
            self.py_r = torch.nn.Parameter(torch.ones(self.n_input_proteins, n_batch))
        elif self.protein_dispersion == "protein-label":
            self.py_r = torch.nn.Parameter(torch.ones(self.n_input_proteins, n_labels))
        else:  # protein-cell
            pass

        # z encoder goes from the n_input-dimensional data to an n_latent-d
        # latent space representation
        self.encoder = EncoderTOTALVI(
            n_input_genes + self.n_input_proteins,
            n_latent,
            n_layers=n_layers_encoder,
            n_cat_list=[n_batch] if encoder_batch else None,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_encoder,
            distribution=latent_distribution,
        )
        self.decoder = DecoderTOTALVI(
            n_latent,
            n_input_genes,
            self.n_input_proteins,
            n_layers=n_layers_decoder,
            n_cat_list=[n_batch],
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_decoder,
        )

    def sample_from_posterior_z(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        batch_index: Optional[torch.Tensor] = None,
        give_mean: bool = False,
        n_samples: int = 5000,
    ) -> torch.Tensor:
        """ Access the tensor of latent values from the posterior

        :param x: tensor of values with shape ``(batch_size, n_input_genes)``
        :param y: tensor of values with shape ``(batch_size, n_input_proteins)``
        :param batch_index: tensor of batch indices
        :param give_mean: Whether to sample, or give mean of distribution
        :return: tensor of shape ``(batch_size, n_latent)``
        """
        if self.log_variational:
            x = torch.log(1.0 + x)
            y = torch.log(1.0 + y)
        qz_m, qz_v, _, _, latent, _ = self.encoder(
            torch.cat((x, y), dim=-1), batch_index
        )
        z = latent["z"]
        if give_mean:
            if self.latent_distribution == "ln":
                samples = Normal(qz_m, qz_v.sqrt()).sample([n_samples])
                z = self.encoder.z_transformation(samples)
                z = z.mean(dim=0)
            else:
                z = qz_m
        return z

    def sample_from_posterior_l(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        batch_index: Optional[torch.Tensor] = None,
        give_mean: bool = True,
    ) -> torch.Tensor:
        """ Provides the tensor of library size from the posterior

        :param x: tensor of values with shape ``(batch_size, n_input_genes)``
        :param y: tensor of values with shape ``(batch_size, n_input_proteins)``
        :return: tensor of shape ``(batch_size, 1)``
        """
        if self.log_variational:
            x = torch.log(1.0 + x)
            y = torch.log(1.0 + y)
        _, _, ql_m, ql_v, latent, _ = self.encoder(
            torch.cat((x, y), dim=-1), batch_index
        )
        library_gene = latent["l"]
        if give_mean is True:
            return torch.exp(ql_m + 0.5 * ql_v)
        else:
            return library_gene

    def get_sample_rate(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        batch_index: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
        n_samples: int = 1,
    ) -> torch.Tensor:
        """Returns the tensor of negative binomial mean for genes

        :param x: tensor of values with shape ``(batch_size, n_input_genes)``
        :param y: tensor of values with shape ``(batch_size, n_input_proteins)``
        :param batch_index: array that indicates which batch the cells belong to with shape ``batch_size``
        :param label: tensor of cell-types labels with shape ``(batch_size, n_labels)``
        :param n_samples: number of samples
        :return: tensor of means of the negative binomial distribution with shape ``(batch_size, n_input_genes)``
        """
        outputs = self.inference(
            x, y, batch_index=batch_index, label=label, n_samples=n_samples
        )
        rate = outputs["px_"]["rate"]
        return rate

    def get_sample_dispersion(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        batch_index: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
        n_samples: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the tensors of dispersions for genes and proteins

        :param x: tensor of values with shape ``(batch_size, n_input_genes)``
        :param y: tensor of values with shape ``(batch_size, n_input_proteins)``
        :param batch_index: array that indicates which batch the cells belong to with shape ``batch_size``
        :param label: tensor of cell-types labels with shape ``(batch_size, n_labels)``
        :param n_samples: number of samples
        :return: tensors of dispersions of the negative binomial distribution
        """
        outputs = self.inference(
            x, y, batch_index=batch_index, label=label, n_samples=n_samples
        )
        px_r = outputs["px_"]["r"]
        py_r = outputs["py_"]["r"]
        return px_r, py_r

    def get_sample_scale(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        batch_index: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
        n_samples: int = 1,
        transform_batch: Optional[int] = None,
        eps=0,
        normalize_pro=False,
        sample_bern=True,
        include_bg=False,
    ) -> torch.Tensor:
        """ Returns tuple of gene and protein scales.

        These scales can also be transformed into a particular batch. This function is
        the core of differential expression.

        :param transform_batch: Int of batch to "transform" all cells into
        :param eps: Prior count to add to protein normalized expression
        :param normalize_pro: bool, whether to make protein expression sum to one in a cell
        :param include_bg: bool, whether to include the background component of expression
        """
        outputs = self.inference(
            x,
            y,
            batch_index=batch_index,
            label=label,
            n_samples=n_samples,
            transform_batch=transform_batch,
        )
        px_ = outputs["px_"]
        py_ = outputs["py_"]
        protein_mixing = 1 / (1 + torch.exp(-py_["mixing"]))
        if sample_bern is True:
            protein_mixing = Bernoulli(protein_mixing).sample()
        pro_value = (1 - protein_mixing) * py_["rate_fore"]
        if include_bg is True:
            pro_value = (1 - protein_mixing) * py_["rate_fore"] + protein_mixing * py_[
                "rate_back"
            ]
        if normalize_pro is True:
            pro_value = torch.nn.functional.normalize(pro_value, p=1, dim=-1)

        return px_["scale"], pro_value + eps

    def get_reconstruction_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        px_: Dict[str, torch.Tensor],
        py_: Dict[str, torch.Tensor],
        pro_batch_mask_minibatch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute reconstruction loss
        """
        # Reconstruction Loss
        if self.reconstruction_loss_gene == "zinb":
            reconst_loss_gene = (
                -ZeroInflatedNegativeBinomial(
                    mu=px_["rate"], theta=px_["r"], zi_logits=px_["dropout"]
                )
                .log_prob(x)
                .sum(dim=-1)
            )
        else:
            reconst_loss_gene = (
                -NegativeBinomial(mu=px_["rate"], theta=px_["r"])
                .log_prob(x)
                .sum(dim=-1)
            )

        reconst_loss_protein_full = -log_mixture_nb(
            y, py_["rate_back"], py_["rate_fore"], py_["r"], None, py_["mixing"]
        )
        if pro_batch_mask_minibatch is not None:
            temp_pro_loss_full = torch.zeros_like(reconst_loss_protein_full)
            temp_pro_loss_full.masked_scatter_(
                pro_batch_mask_minibatch.bool(), reconst_loss_protein_full
            )

            reconst_loss_protein = temp_pro_loss_full.sum(dim=-1)
        else:
            reconst_loss_protein = reconst_loss_protein_full.sum(dim=-1)

        return reconst_loss_gene, reconst_loss_protein

    def inference(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        batch_index: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
        n_samples=1,
        transform_batch: Optional[int] = None,
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """ Internal helper function to compute necessary inference quantities

         We use the dictionary ``px_`` to contain the parameters of the ZINB/NB for genes.
         The rate refers to the mean of the NB, dropout refers to Bernoulli mixing parameters.
         `scale` refers to the quanity upon which differential expression is performed. For genes,
         this can be viewed as the mean of the underlying gamma distribution.

         We use the dictionary ``py_`` to contain the parameters of the Mixture NB distribution for proteins.
         `rate_fore` refers to foreground mean, while `rate_back` refers to background mean. ``scale`` refers to
         foreground mean adjusted for background probability and scaled to reside in simplex.
         ``back_alpha`` and ``back_beta`` are the posterior parameters for ``rate_back``.  ``fore_scale`` is the scaling
         factor that enforces `rate_fore` > `rate_back`.

         ``px_["r"]`` and ``py_["r"]`` are the inverse dispersion parameters for genes and protein, respectively.
        """
        x_ = x
        y_ = y
        if self.log_variational:
            x_ = torch.log(1.0 + x_)
            y_ = torch.log(1.0 + y_)

        # Sampling - Encoder gets concatenated genes + proteins
        qz_m, qz_v, ql_m, ql_v, latent, untran_latent = self.encoder(
            torch.cat((x_, y_), dim=-1), batch_index
        )
        z = latent["z"]
        
        library_gene = latent["l"]
        untran_z = untran_latent["z"]
        untran_l = untran_latent["l"]

        if n_samples > 1:
            qz_m = qz_m.unsqueeze(0).expand((n_samples, qz_m.size(0), qz_m.size(1)))
            qz_v = qz_v.unsqueeze(0).expand((n_samples, qz_v.size(0), qz_v.size(1)))
            untran_z = Normal(qz_m, qz_v.sqrt()).sample()
            z = self.encoder.z_transformation(untran_z)
            ql_m = ql_m.unsqueeze(0).expand((n_samples, ql_m.size(0), ql_m.size(1)))
            ql_v = ql_v.unsqueeze(0).expand((n_samples, ql_v.size(0), ql_v.size(1)))
            untran_l = Normal(ql_m, ql_v.sqrt()).sample()
            library_gene = self.encoder.l_transformation(untran_l)

        if self.gene_dispersion == "gene-label":
            # px_r gets transposed - last dimension is nb genes
            px_r = F.linear(one_hot(label, self.n_labels), self.px_r)
        elif self.gene_dispersion == "gene-batch":
            px_r = F.linear(one_hot(batch_index, self.n_batch), self.px_r)
        elif self.gene_dispersion == "gene":
            px_r = self.px_r
        px_r = torch.exp(px_r)

        if self.protein_dispersion == "protein-label":
            # py_r gets transposed - last dimension is n_proteins
            py_r = F.linear(one_hot(label, self.n_labels), self.py_r)
        elif self.protein_dispersion == "protein-batch":
            py_r = F.linear(one_hot(batch_index, self.n_batch), self.py_r)
        elif self.protein_dispersion == "protein":
            py_r = self.py_r
        py_r = torch.exp(py_r)

        # Background regularization
        if self.n_batch > 0:
            py_back_alpha_prior = F.linear(
                one_hot(batch_index, self.n_batch), self.background_pro_alpha
            )
            py_back_beta_prior = F.linear(
                one_hot(batch_index, self.n_batch),
                torch.exp(self.background_pro_log_beta),
            )
        else:
            py_back_alpha_prior = self.background_pro_alpha
            py_back_beta_prior = torch.exp(self.background_pro_log_beta)
        self.back_mean_prior = Normal(py_back_alpha_prior, py_back_beta_prior)

        if transform_batch is not None:
            batch_index = torch.ones_like(batch_index) * transform_batch
        px_, py_, log_pro_back_mean = self.decoder(z, library_gene, batch_index, label)
        px_["r"] = px_r
        py_["r"] = py_r

        return dict(
            px_=px_,
            py_=py_,
            qz_m=qz_m,
            qz_v=qz_v,
            z=z,
            untran_z=untran_z,
            ql_m=ql_m,
            ql_v=ql_v,
            library_gene=library_gene,
            untran_l=untran_l,
            log_pro_back_mean=log_pro_back_mean,
        )

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        local_l_mean_gene: torch.Tensor,
        local_l_var_gene: torch.Tensor,
        batch_index: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor
    ]:
        """ Returns the reconstruction loss and the Kullback divergences

        :param x: tensor of values with shape ``(batch_size, n_input_genes)``
        :param y: tensor of values with shape ``(batch_size, n_input_proteins)``
        :param local_l_mean_gene: tensor of means of the prior distribution of latent variable l
         with shape ``(batch_size, 1)````
        :param local_l_var_gene: tensor of variancess of the prior distribution of latent variable l
         with shape ``(batch_size, 1)``
        :param batch_index: array that indicates which batch the cells belong to with shape ``batch_size``
        :param label: tensor of cell-types labels with shape (batch_size, n_labels)
        :return: the reconstruction loss and the Kullback divergences
        """
        # Parameters for z latent distribution
        outputs = self.inference(x, y, batch_index, label)
        qz_m = outputs["qz_m"]
        qz_v = outputs["qz_v"]
        ql_m = outputs["ql_m"]
        ql_v = outputs["ql_v"]
        px_ = outputs["px_"]
        py_ = outputs["py_"]

        if self.protein_batch_mask is not None:
            pro_batch_mask_minibatch = torch.zeros_like(y)
            for b in np.arange(len(torch.unique(batch_index))):
                b_indices = (batch_index == b).reshape(-1)
                pro_batch_mask_minibatch[b_indices] = torch.tensor(
                    self.protein_batch_mask[b].astype(np.float32), device=y.device
                )
        else:
            pro_batch_mask_minibatch = None

        reconst_loss_gene, reconst_loss_protein = self.get_reconstruction_loss(
            x, y, px_, py_, pro_batch_mask_minibatch
        )

        
        # print()

        # KL Divergence
        kl_div_z = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(0, 1)).sum(dim=1)
        kl_div_l_gene = kl(
            Normal(ql_m, torch.sqrt(ql_v)),
            Normal(local_l_mean_gene, torch.sqrt(local_l_var_gene)),
        ).sum(dim=1)

        kl_div_back_pro_full = kl(
            Normal(py_["back_alpha"], py_["back_beta"]), self.back_mean_prior
        )
        if pro_batch_mask_minibatch is not None:
            kl_div_back_pro = (pro_batch_mask_minibatch * kl_div_back_pro_full).sum(
                dim=1
            )
        else:
            kl_div_back_pro = kl_div_back_pro_full.sum(dim=1)

        return (
            reconst_loss_gene,
            reconst_loss_protein,
            kl_div_z,
            kl_div_l_gene,
            kl_div_back_pro,
        )


class TOTALVI_shared(nn.Module):
    """Total variational inference for CITE-seq data

    Implements the totalVI model of [Gayoso19]_.

    :param n_input_genes: Number of input genes
    :param n_input_proteins: Number of input proteins
    :param n_batch: Number of batches
    :param n_labels: Number of labels
    :param n_hidden: Number of nodes per hidden layer for the z encoder (protein+genes),
                     genes library encoder, z->genes+proteins decoder
    :param n_latent: Dimensionality of the latent space
    :param n_layers: Number of hidden layers used for encoder and decoder NNs
    :param dropout_rate: Dropout rate for neural networks
    :param genes_dispersion: One of the following

        * ``'gene'`` - genes_dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - genes_dispersion can differ between different batches
        * ``'gene-label'`` - genes_dispersion can differ between different labels

    :param protein_dispersion: One of the following

        * ``'protein'`` - protein_dispersion parameter is constant per protein across cells
        * ``'protein-batch'`` - protein_dispersion can differ between different batches NOT TESTED
        * ``'protein-label'`` - protein_dispersion can differ between different labels NOT TESTED

    :param log_variational: Log(data+1) prior to encoding for numerical stability. Not normalization.
    :param reconstruction_loss_genes:  One of

        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution

    :param latent_distribution:  One of

        * ``'normal'`` - Isotropic normal
        * ``'ln'`` - Logistic normal with normal params N(0, 1)

    Examples:
        >>> dataset = Dataset10X(dataset_name="pbmc_10k_protein_v3", save_path=save_path)
        >>> totalvae = TOTALVI(gene_dataset.nb_genes, len(dataset.protein_names), use_cuda=True)
    """

    def __init__(
        self,
        n_input_shared: int,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 256,
        n_latent: int = 20,
        n_layers_encoder: int = 1,
        n_layers_decoder: int = 1,
        dropout_rate_decoder: float = 0.2,
        dropout_rate_encoder: float = 0.2,
        gene_dispersion: str = "gene",
        protein_dispersion: str = "protein",
        log_variational: bool = True,
        reconstruction_loss_gene: str = "nb",
        latent_distribution: str = "normal",
        protein_batch_mask: List[np.ndarray] = None,
        encoder_batch: bool = True,
    ):
        super().__init__()
        self.gene_dispersion = gene_dispersion
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.reconstruction_loss_gene = reconstruction_loss_gene
        self.n_batch = n_batch
        self.n_labels = n_labels
        self.protein_dispersion = protein_dispersion
        self.latent_distribution = latent_distribution
        self.protein_batch_mask = protein_batch_mask

        # parameters for prior on rate_back (background protein mean)
        if n_batch > 0:
            self.background_pro_alpha = torch.nn.Parameter(
                torch.randn(n_input_shared, n_batch)
            )
            self.background_pro_log_beta = torch.nn.Parameter(
                torch.clamp(torch.randn(n_input_shared, n_batch), -10, 1)
            )
        else:
            self.background_pro_alpha = torch.nn.Parameter(
                torch.randn(n_input_shared)
            )
            self.background_pro_log_beta = torch.nn.Parameter(
                torch.clamp(torch.randn(n_input_shared), -10, 1)
            )

        if self.gene_dispersion == "gene":
            self.px_r = torch.nn.Parameter(torch.randn(n_input_shared))
        elif self.gene_dispersion == "gene-batch":
            self.px_r = torch.nn.Parameter(torch.randn(n_input_shared, n_batch))
        elif self.gene_dispersion == "gene-label":
            self.px_r = torch.nn.Parameter(torch.randn(n_input_shared, n_labels))
        else:  # gene-cell
            pass

        if self.protein_dispersion == "protein":
            self.py_r = torch.nn.Parameter(torch.ones(n_input_shared))
        elif self.protein_dispersion == "protein-batch":
            self.py_r = torch.nn.Parameter(torch.ones(n_input_shared, n_batch))
        elif self.protein_dispersion == "protein-label":
            self.py_r = torch.nn.Parameter(torch.ones(n_input_shared, n_labels))
        else:  # protein-cell
            pass

        # z encoder goes from the n_input-dimensional data to an n_latent-d
        # latent space representation
        self.encoder = EncoderTOTALVI(
            n_input_shared+1,
            n_latent,
            n_layers=n_layers_encoder,
            n_cat_list=[n_batch] if encoder_batch else None,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_encoder,
            distribution=latent_distribution,
        )

        self.decoder = DecoderTOTALVI_shared(
            n_latent+1,
            n_input_shared,
            n_input_shared,
            n_layers=n_layers_decoder,
            n_cat_list=[n_batch],
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_decoder,
        )

         

    def sample_from_posterior_z(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        batch_index: Optional[torch.Tensor] = None,
        give_mean: bool = False,
        n_samples: int = 5000,
    ) -> torch.Tensor:
        """ Access the tensor of latent values from the posterior

        :param x: tensor of values with shape ``(batch_size, n_input_genes)``
        :param y: tensor of values with shape ``(batch_size, n_input_proteins)``
        :param batch_index: tensor of batch indices
        :param give_mean: Whether to sample, or give mean of distribution
        :return: tensor of shape ``(batch_size, n_latent)``
        """
        if self.log_variational:
            x = torch.log(1 + x)
            y = torch.log(1 + y)
        qz_m, qz_v, _, _, latent, _ = self.encoder(
            torch.cat((x, y), dim=0), batch_index
        )
        z = latent["z"]
        if give_mean:
            if self.latent_distribution == "ln":
                samples = Normal(qz_m, qz_v.sqrt()).sample([n_samples])
                z = self.encoder.z_transformation(samples)
                z = z.mean(dim=0)
            else:
                z = qz_m
        return z

    def sample_from_posterior_l(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        batch_index: Optional[torch.Tensor] = None,
        give_mean: bool = True,
    ) -> torch.Tensor:
        """ Provides the tensor of library size from the posterior

        :param x: tensor of values with shape ``(batch_size, n_input_genes)``
        :param y: tensor of values with shape ``(batch_size, n_input_proteins)``
        :return: tensor of shape ``(batch_size, 1)``
        """
        if self.log_variational:
            x = torch.log(1 + x)
            y = torch.log(1 + y)
        _, _, ql_m, ql_v, latent, _ = self.encoder(
            torch.cat((x, y), dim=0), batch_index
        )
        library_gene = latent["l"]
        if give_mean is True:
            return torch.exp(ql_m + 0.5 * ql_v)
        else:
            return library_gene

    def get_sample_rate(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        batch_index: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
        n_samples: int = 1,
    ) -> torch.Tensor:
        """Returns the tensor of negative binomial mean for genes

        :param x: tensor of values with shape ``(batch_size, n_input_genes)``
        :param y: tensor of values with shape ``(batch_size, n_input_proteins)``
        :param batch_index: array that indicates which batch the cells belong to with shape ``batch_size``
        :param label: tensor of cell-types labels with shape ``(batch_size, n_labels)``
        :param n_samples: number of samples
        :return: tensor of means of the negative binomial distribution with shape ``(batch_size, n_input_genes)``
        """
        outputs = self.inference(
            x, y, batch_index=batch_index, label=label, n_samples=n_samples
        )
        rate = outputs["px_"]["rate"]
        return rate

    def get_sample_dispersion(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        batch_index: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
        n_samples: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the tensors of dispersions for genes and proteins

        :param x: tensor of values with shape ``(batch_size, n_input_genes)``
        :param y: tensor of values with shape ``(batch_size, n_input_proteins)``
        :param batch_index: array that indicates which batch the cells belong to with shape ``batch_size``
        :param label: tensor of cell-types labels with shape ``(batch_size, n_labels)``
        :param n_samples: number of samples
        :return: tensors of dispersions of the negative binomial distribution
        """
        outputs = self.inference(
            x, y, batch_index=batch_index, label=label, n_samples=n_samples
        )
        px_r = outputs["px_"]["r"]
        py_r = outputs["py_"]["r"]
        return px_r, py_r

    def get_sample_scale(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        batch_index: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
        n_samples: int = 1,
        transform_batch: Optional[int] = None,
        eps=0,
        normalize_pro=False,
        sample_bern=True,
        include_bg=False,
    ) -> torch.Tensor:
        """ Returns tuple of gene and protein scales.

        These scales can also be transformed into a particular batch. This function is
        the core of differential expression.

        :param transform_batch: Int of batch to "transform" all cells into
        :param eps: Prior count to add to protein normalized expression
        :param normalize_pro: bool, whether to make protein expression sum to one in a cell
        :param include_bg: bool, whether to include the background component of expression
        """
        outputs = self.inference(
            x,
            y,
            batch_index=batch_index,
            label=label,
            n_samples=n_samples,
            transform_batch=transform_batch,
        )
        px_ = outputs["px_"]
        py_ = outputs["py_"]
        protein_mixing = 1 / (1 + torch.exp(-py_["mixing"]))
        if sample_bern is True:
            protein_mixing = Bernoulli(protein_mixing).sample()
        pro_value = (1 - protein_mixing) * py_["rate_fore"]
        if include_bg is True:
            pro_value = (1 - protein_mixing) * py_["rate_fore"] + protein_mixing * py_[
                "rate_back"
            ]
        if normalize_pro is True:
            pro_value = torch.nn.functional.normalize(pro_value, p=1, dim=-1)

        return px_["scale"], pro_value + eps

    def get_reconstruction_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        px_: Dict[str, torch.Tensor],
        py_: Dict[str, torch.Tensor],
        pro_batch_mask_minibatch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute reconstruction loss
        """
        # Reconstruction Loss
        if self.reconstruction_loss_gene == "zinb":
            reconst_loss_gene = (
                -ZeroInflatedNegativeBinomial(
                    mu=px_["rate"], theta=px_["r"], zi_logits=px_["dropout"]
                )
                .log_prob(x)
                .sum(dim=-1)
            )
        else:
            reconst_loss_gene = (
                -NegativeBinomial(mu=px_["rate"], theta=px_["r"])
                .log_prob(x)
                .sum(dim=-1)
            )

        reconst_loss_protein_full = -log_mixture_nb(
            y, py_["rate_back"], py_["rate_fore"], py_["r"], None, py_["mixing"]
        )
        if pro_batch_mask_minibatch is not None:
            temp_pro_loss_full = torch.zeros_like(reconst_loss_protein_full)
            temp_pro_loss_full.masked_scatter_(
                pro_batch_mask_minibatch.bool(), reconst_loss_protein_full
            )

            reconst_loss_protein = temp_pro_loss_full.sum(dim=-1)
        else:
            reconst_loss_protein = reconst_loss_protein_full.sum(dim=-1)

        return reconst_loss_gene, reconst_loss_protein

    def inference(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        batch_index: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
        n_samples=1,
        transform_batch: Optional[int] = None,
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """ Internal helper function to compute necessary inference quantities

         We use the dictionary ``px_`` to contain the parameters of the ZINB/NB for genes.
         The rate refers to the mean of the NB, dropout refers to Bernoulli mixing parameters.
         `scale` refers to the quanity upon which differential expression is performed. For genes,
         this can be viewed as the mean of the underlying gamma distribution.

         We use the dictionary ``py_`` to contain the parameters of the Mixture NB distribution for proteins.
         `rate_fore` refers to foreground mean, while `rate_back` refers to background mean. ``scale`` refers to
         foreground mean adjusted for background probability and scaled to reside in simplex.
         ``back_alpha`` and ``back_beta`` are the posterior parameters for ``rate_back``.  ``fore_scale`` is the scaling
         factor that enforces `rate_fore` > `rate_back`.

         ``px_["r"]`` and ``py_["r"]`` are the inverse dispersion parameters for genes and protein, respectively.
        """
        x_ = x
        y_ = y
        
        x_size = x.shape[0]

        if self.log_variational:
            x_ = torch.log(1 + x_)
            y_ = torch.log(1 + y_)

        # Sampling - Encoder gets concatenated genes + proteins
        additional_columns = torch.tensor([[0.0]] * x_.shape[0], dtype=torch.float32,device='cuda')
        x_extended = torch.cat([x_, additional_columns], dim=1)
        additional_columns = torch.tensor([[1.0]] * y_.shape[0], dtype=torch.float32,device='cuda')
        y_extended = torch.cat([y_, additional_columns], dim=1)
        orig_tensor = torch.cat((x_extended, y_extended), dim=0)

        # permuted_indices = torch.randperm(orig_tensor.size(0))
        # orig_tensor_perm = orig_tensor[permuted_indices]

        # qz_m, qz_v, ql_m, ql_v, latent, untran_latent = self.encoder(
        #     orig_tensor_perm, batch_index
        # )
        qz_m, qz_v, ql_m, ql_v, latent, untran_latent = self.encoder(
            orig_tensor, batch_index
        )


        # qz_m = qz_m[torch.argsort(permuted_indices)]
        # qz_v = qz_v[torch.argsort(permuted_indices)]
        # ql_m = ql_m[torch.argsort(permuted_indices)]
        # ql_v = ql_v[torch.argsort(permuted_indices)]

        # latent["z"] = latent["z"][torch.argsort(permuted_indices)]
        # latent["l"] = latent["l"][torch.argsort(permuted_indices)]
        # untran_latent["z"] = untran_latent["z"][torch.argsort(permuted_indices)]
        # untran_latent["l"] = untran_latent["l"][torch.argsort(permuted_indices)]
        

        z = latent["z"]
        gene_size = x_.shape[0]
        z_gene = z[:gene_size]
        z_protein = z[gene_size:]       
        
        additional_columns = torch.tensor([[0.0]] * x_.shape[0], dtype=torch.float32,device='cuda')
        z_gene = torch.cat([z_gene, additional_columns], dim=1)
        additional_columns = torch.tensor([[1.0]] * y_.shape[0], dtype=torch.float32,device='cuda')
        z_protein = torch.cat([z_protein, additional_columns], dim=1)
        z = torch.cat((z_gene, z_protein), dim=0)
        
        library_gene = latent["l"]
        untran_z = untran_latent["z"]
        untran_l = untran_latent["l"]

        if n_samples > 1:
            qz_m = qz_m.unsqueeze(0).expand((n_samples, qz_m.size(0), qz_m.size(1)))
            qz_v = qz_v.unsqueeze(0).expand((n_samples, qz_v.size(0), qz_v.size(1)))
            untran_z = Normal(qz_m, qz_v.sqrt()).sample()
            z = self.encoder.z_transformation(untran_z)
            ql_m = ql_m.unsqueeze(0).expand((n_samples, ql_m.size(0), ql_m.size(1)))
            ql_v = ql_v.unsqueeze(0).expand((n_samples, ql_v.size(0), ql_v.size(1)))
            untran_l = Normal(ql_m, ql_v.sqrt()).sample()
            library_gene = self.encoder.l_transformation(untran_l)


        if self.gene_dispersion == "gene-label":
            # px_r gets transposed - last dimension is nb genes
            px_r = F.linear(one_hot(label, self.n_labels), self.px_r)
        elif self.gene_dispersion == "gene-batch":
            px_r = F.linear(one_hot(batch_index, self.n_batch), self.px_r)
        elif self.gene_dispersion == "gene":
            px_r = self.px_r
        px_r = torch.exp(px_r)

        if self.protein_dispersion == "protein-label":
            # py_r gets transposed - last dimension is n_proteins
            py_r = F.linear(one_hot(label, self.n_labels), self.py_r)
        elif self.protein_dispersion == "protein-batch":
            py_r = F.linear(one_hot(batch_index, self.n_batch), self.py_r)
        elif self.protein_dispersion == "protein":
            py_r = self.py_r
        py_r = torch.exp(py_r)

        # Background regularization
        if self.n_batch > 0:
            py_back_alpha_prior = F.linear(
                one_hot(batch_index, self.n_batch), self.background_pro_alpha
            )
            py_back_beta_prior = F.linear(
                one_hot(batch_index, self.n_batch),
                torch.exp(self.background_pro_log_beta),
            )
        else:
            py_back_alpha_prior = self.background_pro_alpha
            py_back_beta_prior = torch.exp(self.background_pro_log_beta)

        self.back_mean_prior = Normal(py_back_alpha_prior, py_back_beta_prior)

        if transform_batch is not None:
            batch_index = torch.ones_like(batch_index) * transform_batch
        

        px_, py_, log_pro_back_mean = self.decoder(x_size,z,library_gene, batch_index, label)
        px_["r"] = px_r
        py_["r"] = py_r

        return dict(
            px_=px_,
            py_=py_,
            qz_m=qz_m,
            qz_v=qz_v,
            z=z,
            untran_z=untran_z,
            ql_m=ql_m,
            ql_v=ql_v,
            library_gene=library_gene,
            untran_l=untran_l,
            log_pro_back_mean=log_pro_back_mean,
        )


    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        local_l_mean_gene: torch.Tensor,
        local_l_var_gene: torch.Tensor,
        batch_index: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor
    ]:
        """ Returns the reconstruction loss and the Kullback divergences

        :param x: tensor of values with shape ``(batch_size, n_input_genes)``
        :param y: tensor of values with shape ``(batch_size, n_input_proteins)``
        :param local_l_mean_gene: tensor of means of the prior distribution of latent variable l
         with shape ``(batch_size, 1)````
        :param local_l_var_gene: tensor of variancess of the prior distribution of latent variable l
         with shape ``(batch_size, 1)``
        :param batch_index: array that indicates which batch the cells belong to with shape ``batch_size``
        :param label: tensor of cell-types labels with shape (batch_size, n_labels)
        :return: the reconstruction loss and the Kullback divergences
        """
        # Parameters for z latent distribution

        outputs = self.inference(x, y, batch_index, label)
        qz_m = outputs["qz_m"]
        qz_v = outputs["qz_v"]
        ql_m = outputs["ql_m"]
        ql_v = outputs["ql_v"]
        px_ = outputs["px_"]
        py_ = outputs["py_"]
        z_ = outputs["z"]

        z_gene = z_[:x.shape[0]]
        z_protein = z_[x.shape[0]:]

        if self.protein_batch_mask is not None:
            pro_batch_mask_minibatch = torch.zeros_like(y)
            for b in np.arange(len(torch.unique(batch_index))):
                b_indices = (batch_index == b).reshape(-1)
                pro_batch_mask_minibatch[b_indices] = torch.tensor(
                    self.protein_batch_mask[b].astype(np.float32), device=y.device
                )
        else:
            pro_batch_mask_minibatch = None

        # print(x.shape)
        # print("------\n")
        # print(px_["rate"].shape)
        # print(px_["r"].shape)
        # print(px_["dropout"].shape)
        # print("-------\n")
        # print(ql_v.shape)

        reconst_loss_gene, reconst_loss_protein = self.get_reconstruction_loss(
            x, y, px_, py_, pro_batch_mask_minibatch
        )

        
        # KL Divergence
        kl_div_z = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(0, 1)).sum(dim=1)

        kl_div_l_gene = kl(
            Normal(ql_m[:x.shape[0]], torch.sqrt(ql_v[:x.shape[0]])),
            Normal(local_l_mean_gene, torch.sqrt(local_l_var_gene)),
        ).sum(dim=1)

        kl_div_back_pro_full = kl(
            Normal(py_["back_alpha"], py_["back_beta"]), self.back_mean_prior
        )
        
        if pro_batch_mask_minibatch is not None:
            kl_div_back_pro = (pro_batch_mask_minibatch * kl_div_back_pro_full).sum(
                dim=1
            )
        else:
            kl_div_back_pro = kl_div_back_pro_full.sum(dim=1)

        return (
            reconst_loss_gene,
            reconst_loss_protein,
            kl_div_z,
            kl_div_l_gene,
            kl_div_back_pro,
            z_gene,
            z_protein,
        )


class TOTALVI_shared_sirexp(nn.Module):
    """Total variational inference for CITE-seq data

    Implements the totalVI model of [Gayoso19]_.

    :param n_input_genes: Number of input genes
    :param n_input_proteins: Number of input proteins
    :param n_batch: Number of batches
    :param n_labels: Number of labels
    :param n_hidden: Number of nodes per hidden layer for the z encoder (protein+genes),
                     genes library encoder, z->genes+proteins decoder
    :param n_latent: Dimensionality of the latent space
    :param n_layers: Number of hidden layers used for encoder and decoder NNs
    :param dropout_rate: Dropout rate for neural networks
    :param genes_dispersion: One of the following

        * ``'gene'`` - genes_dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - genes_dispersion can differ between different batches
        * ``'gene-label'`` - genes_dispersion can differ between different labels

    :param protein_dispersion: One of the following

        * ``'protein'`` - protein_dispersion parameter is constant per protein across cells
        * ``'protein-batch'`` - protein_dispersion can differ between different batches NOT TESTED
        * ``'protein-label'`` - protein_dispersion can differ between different labels NOT TESTED

    :param log_variational: Log(data+1) prior to encoding for numerical stability. Not normalization.
    :param reconstruction_loss_genes:  One of

        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution

    :param latent_distribution:  One of

        * ``'normal'`` - Isotropic normal
        * ``'ln'`` - Logistic normal with normal params N(0, 1)

    Examples:
        >>> dataset = Dataset10X(dataset_name="pbmc_10k_protein_v3", save_path=save_path)
        >>> totalvae = TOTALVI(gene_dataset.nb_genes, len(dataset.protein_names), use_cuda=True)
    """

    def __init__(
        self,
        n_input_shared: int,
        n_input_rna_unshared: int,
        n_input_protein_unshared: int,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 256,
        n_latent: int = 20,
        n_layers_encoder: int = 1,
        n_layers_decoder: int = 1,
        dropout_rate_decoder: float = 0.2,
        dropout_rate_encoder: float = 0.2,
        gene_dispersion: str = "gene",
        protein_dispersion: str = "protein",
        log_variational: bool = True,
        reconstruction_loss_gene: str = "nb",
        latent_distribution: str = "normal",
        protein_batch_mask: List[np.ndarray] = None,
        encoder_batch: bool = True,
    ):
        super().__init__()
        self.gene_dispersion = gene_dispersion
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.reconstruction_loss_gene = reconstruction_loss_gene
        self.n_batch = n_batch
        self.n_labels = n_labels
        self.protein_dispersion = protein_dispersion
        self.latent_distribution = latent_distribution
        self.protein_batch_mask = protein_batch_mask

        # parameters for prior on rate_back (background protein mean)
        if n_batch > 0:
            self.background_pro_alpha = torch.nn.Parameter(
                torch.randn(n_input_shared+n_input_protein_unshared, n_batch)
            )
            self.background_pro_log_beta = torch.nn.Parameter(
                torch.clamp(torch.randn(n_input_shared+n_input_protein_unshared, n_batch), -10, 1)
            )
        else:
            self.background_pro_alpha = torch.nn.Parameter(
                torch.randn(n_input_shared + n_input_protein_unshared)
            )
            self.background_pro_log_beta = torch.nn.Parameter(
                torch.clamp(torch.randn(n_input_shared + n_input_protein_unshared), -10, 1)
            )

        if self.gene_dispersion == "gene":
            self.px_r = torch.nn.Parameter(torch.randn(n_input_shared+n_input_rna_unshared))
        elif self.gene_dispersion == "gene-batch":
            self.px_r = torch.nn.Parameter(torch.randn(n_input_shared+n_input_rna_unshared, n_batch))
        elif self.gene_dispersion == "gene-label":
            self.px_r = torch.nn.Parameter(torch.randn(n_input_shared+n_input_rna_unshared, n_labels))
        else:  # gene-cell
            pass

        if self.protein_dispersion == "protein":
            self.py_r = torch.nn.Parameter(torch.ones(n_input_shared+n_input_protein_unshared))
        elif self.protein_dispersion == "protein-batch":
            self.py_r = torch.nn.Parameter(torch.ones(n_input_shared+n_input_protein_unshared, n_batch))
        elif self.protein_dispersion == "protein-label":
            self.py_r = torch.nn.Parameter(torch.ones(n_input_shared+n_input_protein_unshared, n_labels))
        else:  # protein-cell
            pass


        self.shared_encoder = FCLayers(
            n_in=n_input_shared+1,
            n_out=n_hidden,
            n_cat_list=[n_batch] if encoder_batch else None,
            n_layers=n_layers_encoder,
            n_hidden=n_hidden,
            dropout_rate=0.1,
        )

        self.rna_unshared_encoder = FCLayers(
            n_in=n_input_rna_unshared+1,
            n_out=n_hidden,
            n_cat_list=[n_batch] if encoder_batch else None,
            n_layers=n_layers_encoder,
            n_hidden=n_hidden,
            dropout_rate=0.1,
        )


        self.protein_unshared_encoder = FCLayers(
            n_in=n_input_protein_unshared+1,
            n_out=n_hidden,
            n_cat_list=[n_batch] if encoder_batch else None,
            n_layers=n_layers_encoder,
            n_hidden=n_hidden,
            dropout_rate=0.1,
        )

        self.middle =  FCLayers(
            n_in = 3*n_hidden,
            n_out=n_hidden,
            n_cat_list=[n_batch] if encoder_batch else None,
            n_layers=n_layers_encoder,
            n_hidden=n_latent,
            dropout_rate=0.1,
        )


        self.z_encoder = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Dropout(p = 0.1),
        )
        self.z_mean_encoder = nn.Linear(n_hidden, n_latent)
        self.z_var_encoder = nn.Linear(n_hidden, n_latent)

        self.l_gene_encoder = nn.Sequential(
            nn.Linear(n_hidden,n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Dropout(p=0.1),
        )
        self.l_gene_mean_encoder = nn.Linear(n_hidden, 1)
        self.l_gene_var_encoder = nn.Linear(n_hidden, 1)

        self.decoder = DecoderTOTALVI_shared(
            n_latent+1,
            n_input_shared+n_input_rna_unshared,
            n_input_shared+n_input_protein_unshared,
            n_layers=n_layers_decoder,
            n_cat_list=[n_batch],
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_decoder,
        )

         

    def sample_from_posterior_z(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        batch_index: Optional[torch.Tensor] = None,
        give_mean: bool = False,
        n_samples: int = 5000,
    ) -> torch.Tensor:
        """ Access the tensor of latent values from the posterior

        :param x: tensor of values with shape ``(batch_size, n_input_genes)``
        :param y: tensor of values with shape ``(batch_size, n_input_proteins)``
        :param batch_index: tensor of batch indices
        :param give_mean: Whether to sample, or give mean of distribution
        :return: tensor of shape ``(batch_size, n_latent)``
        """
        if self.log_variational:
            x = torch.log(1 + x)
            y = torch.log(1 + y)
        qz_m, qz_v, _, _, latent, _ = self.encoder(
            torch.cat((x, y), dim=0), batch_index
        )
        z = latent["z"]
        if give_mean:
            if self.latent_distribution == "ln":
                samples = Normal(qz_m, qz_v.sqrt()).sample([n_samples])
                z = self.encoder.z_transformation(samples)
                z = z.mean(dim=0)
            else:
                z = qz_m
        return z

    def sample_from_posterior_l(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        batch_index: Optional[torch.Tensor] = None,
        give_mean: bool = True,
    ) -> torch.Tensor:
        """ Provides the tensor of library size from the posterior

        :param x: tensor of values with shape ``(batch_size, n_input_genes)``
        :param y: tensor of values with shape ``(batch_size, n_input_proteins)``
        :return: tensor of shape ``(batch_size, 1)``
        """
        if self.log_variational:
            x = torch.log(1 + x)
            y = torch.log(1 + y)
        _, _, ql_m, ql_v, latent, _ = self.encoder(
            torch.cat((x, y), dim=0), batch_index
        )
        library_gene = latent["l"]
        if give_mean is True:
            return torch.exp(ql_m + 0.5 * ql_v)
        else:
            return library_gene

    def get_sample_rate(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        batch_index: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
        n_samples: int = 1,
    ) -> torch.Tensor:
        """Returns the tensor of negative binomial mean for genes

        :param x: tensor of values with shape ``(batch_size, n_input_genes)``
        :param y: tensor of values with shape ``(batch_size, n_input_proteins)``
        :param batch_index: array that indicates which batch the cells belong to with shape ``batch_size``
        :param label: tensor of cell-types labels with shape ``(batch_size, n_labels)``
        :param n_samples: number of samples
        :return: tensor of means of the negative binomial distribution with shape ``(batch_size, n_input_genes)``
        """
        outputs = self.inference(
            x, y, batch_index=batch_index, label=label, n_samples=n_samples
        )
        rate = outputs["px_"]["rate"]
        return rate

    def get_sample_dispersion(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        batch_index: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
        n_samples: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the tensors of dispersions for genes and proteins

        :param x: tensor of values with shape ``(batch_size, n_input_genes)``
        :param y: tensor of values with shape ``(batch_size, n_input_proteins)``
        :param batch_index: array that indicates which batch the cells belong to with shape ``batch_size``
        :param label: tensor of cell-types labels with shape ``(batch_size, n_labels)``
        :param n_samples: number of samples
        :return: tensors of dispersions of the negative binomial distribution
        """
        outputs = self.inference(
            x, y, batch_index=batch_index, label=label, n_samples=n_samples
        )
        px_r = outputs["px_"]["r"]
        py_r = outputs["py_"]["r"]
        return px_r, py_r

    def get_sample_scale(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        batch_index: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
        n_samples: int = 1,
        transform_batch: Optional[int] = None,
        eps=0,
        normalize_pro=False,
        sample_bern=True,
        include_bg=False,
    ) -> torch.Tensor:
        """ Returns tuple of gene and protein scales.

        These scales can also be transformed into a particular batch. This function is
        the core of differential expression.

        :param transform_batch: Int of batch to "transform" all cells into
        :param eps: Prior count to add to protein normalized expression
        :param normalize_pro: bool, whether to make protein expression sum to one in a cell
        :param include_bg: bool, whether to include the background component of expression
        """
        outputs = self.inference(
            x,
            y,
            batch_index=batch_index,
            label=label,
            n_samples=n_samples,
            transform_batch=transform_batch,
        )
        px_ = outputs["px_"]
        py_ = outputs["py_"]
        protein_mixing = 1 / (1 + torch.exp(-py_["mixing"]))
        if sample_bern is True:
            protein_mixing = Bernoulli(protein_mixing).sample()
        pro_value = (1 - protein_mixing) * py_["rate_fore"]
        if include_bg is True:
            pro_value = (1 - protein_mixing) * py_["rate_fore"] + protein_mixing * py_[
                "rate_back"
            ]
        if normalize_pro is True:
            pro_value = torch.nn.functional.normalize(pro_value, p=1, dim=-1)

        return px_["scale"], pro_value + eps

    def get_reconstruction_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        px_: Dict[str, torch.Tensor],
        py_: Dict[str, torch.Tensor],
        pro_batch_mask_minibatch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute reconstruction loss
        """
        # Reconstruction Loss
        if self.reconstruction_loss_gene == "zinb":
            reconst_loss_gene = (
                -ZeroInflatedNegativeBinomial(
                    mu=px_["rate"], theta=px_["r"], zi_logits=px_["dropout"]
                )
                .log_prob(x)
                .sum(dim=-1)
            )
        else:
            reconst_loss_gene = (
                -NegativeBinomial(mu=px_["rate"], theta=px_["r"])
                .log_prob(x)
                .sum(dim=-1)
            )

        reconst_loss_protein_full = -log_mixture_nb(
            y, py_["rate_back"], py_["rate_fore"], py_["r"], None, py_["mixing"]
        )
        if pro_batch_mask_minibatch is not None:
            temp_pro_loss_full = torch.zeros_like(reconst_loss_protein_full)
            temp_pro_loss_full.masked_scatter_(
                pro_batch_mask_minibatch.bool(), reconst_loss_protein_full
            )

            reconst_loss_protein = temp_pro_loss_full.sum(dim=-1)
        else:
            reconst_loss_protein = reconst_loss_protein_full.sum(dim=-1)

        return reconst_loss_gene, reconst_loss_protein

    def reparameterize_transformation(self, mu, var):
        untran_z = Normal(mu, var.sqrt()).rsample()
        z = untran_z
        return z, untran_z

    def inference(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_unshared: torch.Tensor,
        y_unshared: torch.Tensor,
        batch_index: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
        n_samples=1,
        transform_batch: Optional[int] = None,
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """ Internal helper function to compute necessary inference quantities

         We use the dictionary ``px_`` to contain the parameters of the ZINB/NB for genes.
         The rate refers to the mean of the NB, dropout refers to Bernoulli mixing parameters.
         `scale` refers to the quanity upon which differential expression is performed. For genes,
         this can be viewed as the mean of the underlying gamma distribution.

         We use the dictionary ``py_`` to contain the parameters of the Mixture NB distribution for proteins.
         `rate_fore` refers to foreground mean, while `rate_back` refers to background mean. ``scale`` refers to
         foreground mean adjusted for background probability and scaled to reside in simplex.
         ``back_alpha`` and ``back_beta`` are the posterior parameters for ``rate_back``.  ``fore_scale`` is the scaling
         factor that enforces `rate_fore` > `rate_back`.

         ``px_["r"]`` and ``py_["r"]`` are the inverse dispersion parameters for genes and protein, respectively.
        """
        x_ = x
        y_ = y

        x_unshared_ = x_unshared
        y_unshared_ = y_unshared
 
        if self.log_variational:
            x_ = torch.log(1.0 + x_)
            y_ = torch.log(1.0 + y_)
            x_unshared_ = torch.log(1.0 + x_unshared_)
            y_unshared_ = torch.log(1.0 + y_unshared_)
        
        additional_columns = torch.tensor([[0.0]] * x_.shape[0], dtype=torch.float32,device='cuda')
        x_unshared_extended = torch.cat([x_unshared_, additional_columns], dim=1)
        additional_columns = torch.tensor([[1.0]] * y_.shape[0], dtype=torch.float32,device='cuda')
        x_unshared_extended_zeroes = torch.cat([torch.zeros((y_.shape[0],x_unshared_.shape[1]),dtype=torch.float32,device='cuda'),additional_columns],dim=1)
        orig_tensor_x_unshared = torch.cat((x_unshared_extended, x_unshared_extended_zeroes), dim=0)

        additional_columns = torch.tensor([[0.0]] * x_.shape[0], dtype=torch.float32,device='cuda')
        y_unshared_extended_zeroes = torch.cat([torch.zeros((x_.shape[0],y_unshared_.shape[1]),dtype=torch.float32,device='cuda'),additional_columns],dim=1)
        additional_columns = torch.tensor([[1.0]] * y_.shape[0], dtype=torch.float32,device='cuda')
        y_unshared_extended = torch.cat([y_unshared_, additional_columns], dim=1)
        orig_tensor_y_unshared = torch.cat((y_unshared_extended_zeroes,y_unshared_extended), dim=0)
        x_size = x.shape[0]

        additional_columns = torch.tensor([[0.0]] * x_.shape[0], dtype=torch.float32,device='cuda')
        x_extended = torch.cat([x_, additional_columns], dim=1)
        additional_columns = torch.tensor([[1.0]] * y_.shape[0], dtype=torch.float32,device='cuda')
        y_extended = torch.cat([y_, additional_columns], dim=1)
        orig_tensor = torch.cat((x_extended, y_extended), dim=0)

        emb_runshared = self.rna_unshared_encoder(orig_tensor_x_unshared, batch_index)
        emb_shared = self.shared_encoder(orig_tensor, batch_index)
        emb_punshared = self.protein_unshared_encoder(orig_tensor_y_unshared, batch_index)

        concat_emb = torch.cat([emb_runshared,emb_shared,emb_punshared],dim=-1)
        q = self.middle(concat_emb,batch_index)
        qz = self.z_encoder(q)
        qz_m = self.z_mean_encoder(qz)
        qz_v = torch.exp(self.z_var_encoder(qz)) + 1e-4
        z, untran_z = self.reparameterize_transformation(qz_m, qz_v)
        ql_gene = self.l_gene_encoder(q)
        ql_m = self.l_gene_mean_encoder(ql_gene)
        ql_v = torch.exp(self.l_gene_var_encoder(ql_gene)) + 1e-4
        log_library_gene = torch.clamp(reparameterize_gaussian(ql_m, ql_v), max=15)
        library_gene = torch.exp(log_library_gene)
        latent = {}
        untran_latent = {}
        latent["z"] = z
        latent["l"] = library_gene
        untran_latent["z"] = untran_z
        untran_latent["l"] = log_library_gene

        z = latent["z"]
        gene_size = x_.shape[0]
        z_gene = z[:gene_size]
        z_protein = z[gene_size:]       
        additional_columns = torch.tensor([[0.0]] * x_.shape[0], dtype=torch.float32,device='cuda')
        z_gene = torch.cat([z_gene, additional_columns], dim=1)
        additional_columns = torch.tensor([[1.0]] * y_.shape[0], dtype=torch.float32,device='cuda')
        z_protein = torch.cat([z_protein, additional_columns], dim=1)
        z = torch.cat((z_gene, z_protein), dim=0)
        library_gene = latent["l"]
        untran_z = untran_latent["z"]
        untran_l = untran_latent["l"]

        if n_samples > 1:
            qz_m = qz_m.unsqueeze(0).expand((n_samples, qz_m.size(0), qz_m.size(1)))
            qz_v = qz_v.unsqueeze(0).expand((n_samples, qz_v.size(0), qz_v.size(1)))
            untran_z = Normal(qz_m, qz_v.sqrt()).sample()
            z = self.encoder.z_transformation(untran_z)
            ql_m = ql_m.unsqueeze(0).expand((n_samples, ql_m.size(0), ql_m.size(1)))
            ql_v = ql_v.unsqueeze(0).expand((n_samples, ql_v.size(0), ql_v.size(1)))
            untran_l = Normal(ql_m, ql_v.sqrt()).sample()
            library_gene = self.encoder.l_transformation(untran_l)


        if self.gene_dispersion == "gene-label":
            # px_r gets transposed - last dimension is nb genes
            px_r = F.linear(one_hot(label, self.n_labels), self.px_r)
        elif self.gene_dispersion == "gene-batch":
            px_r = F.linear(one_hot(batch_index, self.n_batch), self.px_r)
        elif self.gene_dispersion == "gene":
            px_r = self.px_r
        px_r = torch.exp(px_r)

        if self.protein_dispersion == "protein-label":
            # py_r gets transposed - last dimension is n_proteins
            py_r = F.linear(one_hot(label, self.n_labels), self.py_r)
        elif self.protein_dispersion == "protein-batch":
            py_r = F.linear(one_hot(batch_index, self.n_batch), self.py_r)
        elif self.protein_dispersion == "protein":
            py_r = self.py_r
        py_r = torch.exp(py_r)

        # Background regularization
        if self.n_batch > 0:
            py_back_alpha_prior = F.linear(
                one_hot(batch_index, self.n_batch), self.background_pro_alpha
            )
            py_back_beta_prior = F.linear(
                one_hot(batch_index, self.n_batch),
                torch.exp(self.background_pro_log_beta),
            )
        else:
            py_back_alpha_prior = self.background_pro_alpha
            py_back_beta_prior = torch.exp(self.background_pro_log_beta)

        self.back_mean_prior = Normal(py_back_alpha_prior, py_back_beta_prior)

        if transform_batch is not None:
            batch_index = torch.ones_like(batch_index) * transform_batch
        
        px_, py_, log_pro_back_mean = self.decoder(x_size,z,library_gene,batch_index,label)
        px_["r"] = px_r
        py_["r"] = py_r

        return dict(
            px_=px_,
            py_=py_,
            qz_m=qz_m,
            qz_v=qz_v,
            z=z,
            untran_z=untran_z,
            ql_m=ql_m,
            ql_v=ql_v,
            library_gene=library_gene,
            untran_l=untran_l,
            log_pro_back_mean=log_pro_back_mean,
        )


    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_unshared: torch.Tensor,
        y_unshared: torch.Tensor,
        local_l_mean_gene: torch.Tensor,
        local_l_var_gene: torch.Tensor,
        batch_index: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor
    ]:
        """ Returns the reconstruction loss and the Kullback divergences

        :param x: tensor of values with shape ``(batch_size, n_input_genes)``
        :param y: tensor of values with shape ``(batch_size, n_input_proteins)``
        :param local_l_mean_gene: tensor of means of the prior distribution of latent variable l
         with shape ``(batch_size, 1)````
        :param local_l_var_gene: tensor of variancess of the prior distribution of latent variable l
         with shape ``(batch_size, 1)``
        :param batch_index: array that indicates which batch the cells belong to with shape ``batch_size``
        :param label: tensor of cell-types labels with shape (batch_size, n_labels)
        :return: the reconstruction loss and the Kullback divergences
        """
        # Parameters for z latent distribution

        outputs = self.inference(x,y,x_unshared,y_unshared,batch_index,label)
        qz_m = outputs["qz_m"]
        qz_v = outputs["qz_v"]
        ql_m = outputs["ql_m"]
        ql_v = outputs["ql_v"]
        px_ = outputs["px_"]
        py_ = outputs["py_"]
        z_ = outputs["z"]

        z_gene = z_[:x.shape[0]]
        z_protein = z_[x.shape[0]:]

        if self.protein_batch_mask is not None:
            pro_batch_mask_minibatch = torch.zeros_like(y)
            for b in np.arange(len(torch.unique(batch_index))):
                b_indices = (batch_index == b).reshape(-1)
                pro_batch_mask_minibatch[b_indices] = torch.tensor(
                    self.protein_batch_mask[b].astype(np.float32), device=y.device
                )
        else:
            pro_batch_mask_minibatch = None

        reconst_loss_gene, reconst_loss_protein = self.get_reconstruction_loss(
            torch.cat((x_unshared,x), dim = 1), torch.cat((y_unshared,y), dim = 1), px_, py_, pro_batch_mask_minibatch
        )

        kl_div_z = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(0, 1)).sum(dim=1)

        kl_div_l_gene = kl(
            Normal(ql_m[:x.shape[0]], torch.sqrt(ql_v[:x.shape[0]])),
            Normal(local_l_mean_gene, torch.sqrt(local_l_var_gene)),
        ).sum(dim=1)

        kl_div_back_pro_full = kl(
            Normal(py_["back_alpha"], py_["back_beta"]), self.back_mean_prior
        )
        
        if pro_batch_mask_minibatch is not None:
            kl_div_back_pro = (pro_batch_mask_minibatch * kl_div_back_pro_full).sum(
                dim=1
            )
        else:
            kl_div_back_pro = kl_div_back_pro_full.sum(dim=1)

        return (
            reconst_loss_gene,
            reconst_loss_protein,
            kl_div_z,
            kl_div_l_gene,
            kl_div_back_pro,
            z_gene,
            z_protein,
        )

class ProductOfExperts(nn.Module):
    def forward(self, mu, var, eps=1e-8):
        T         = 1. / var
        pd_mu     = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var    = 1. / torch.sum(T, dim=0)
        # pd_logvar = torch.log(pd_var)
        return pd_mu, pd_var

def build_multi_layers(layers, use_batch_norm=True, dropout_rate = 0.1 ):
    """Build multilayer linear perceptron"""
    if dropout_rate > 0:
        fc_layers = nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        "Layer {}".format(i),
                        nn.Sequential(
                            nn.Linear(n_in, n_out),
                            nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001),
                            nn.ReLU(),
                            nn.Dropout(p=dropout_rate),
                        ),
                    )

                    for i, (n_in, n_out) in enumerate( zip(layers[:-1], layers[1:] ) )
                ]
            )
        )

    else:
        fc_layers = nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        "Layer {}".format(i),
                        nn.Sequential(
                            nn.Linear(n_in, n_out),
                            nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001),
                            nn.ReLU(),
                        ),
                    )

                    for i, (n_in, n_out) in enumerate( zip(layers[:-1], layers[1:] ) )
                ]
            )
        )
        
        
    
    return fc_layers


class TOTALVI_shared_sirPOE(nn.Module):
    """Total variational inference for CITE-seq data
    Implements the totalVI model of [Gayoso19]_.
    :param n_input_genes: Number of input genes
    :param n_input_proteins: Number of input proteins
    :param n_batch: Number of batches
    :param n_labels: Number of labels
    :param n_hidden: Number of nodes per hidden layer for the z encoder (protein+genes),
                     genes library encoder, z->genes+proteins decoder
    :param n_latent: Dimensionality of the latent space
    :param n_layers: Number of hidden layers used for encoder and decoder NNs
    :param dropout_rate: Dropout rate for neural networks
    :param genes_dispersion: One of the following

        * ``'gene'`` - genes_dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - genes_dispersion can differ between different batches
        * ``'gene-label'`` - genes_dispersion can differ between different labels

    :param protein_dispersion: One of the following

        * ``'protein'`` - protein_dispersion parameter is constant per protein across cells
        * ``'protein-batch'`` - protein_dispersion can differ between different batches NOT TESTED
        * ``'protein-label'`` - protein_dispersion can differ between different labels NOT TESTED

    :param log_variational: Log(data+1) prior to encoding for numerical stability. Not normalization.
    :param reconstruction_loss_genes:  One of

        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution

    :param latent_distribution:  One of

        * ``'normal'`` - Isotropic normal
        * ``'ln'`` - Logistic normal with normal params N(0, 1)

    Examples:
        >>> dataset = Dataset10X(dataset_name="pbmc_10k_protein_v3", save_path=save_path)
        >>> totalvae = TOTALVI(gene_dataset.nb_genes, len(dataset.protein_names), use_cuda=True)
    """

    def __init__(
        self,
        n_input_shared: int,
        n_input_rna_unshared: int,
        n_input_protein_unshared: int,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 256,
        n_latent: int = 20,
        n_layers_encoder: int = 1,
        n_layers_decoder: int = 1,
        dropout_rate_decoder: float = 0.1,
        dropout_rate_encoder: float = 0.1,
        gene_dispersion: str = "gene",
        protein_dispersion: str = "protein",
        log_variational: bool = True,
        reconstruction_loss_gene: str = "nb",
        latent_distribution: str = "normal",
        protein_batch_mask: List[np.ndarray] = None,
        encoder_batch: bool = True,
    ):
        super().__init__()
        self.gene_dispersion = gene_dispersion
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.reconstruction_loss_gene = reconstruction_loss_gene
        self.n_batch = n_batch
        self.n_labels = n_labels
        self.protein_dispersion = protein_dispersion
        self.latent_distribution = latent_distribution
        self.protein_batch_mask = protein_batch_mask
        
        self.background_pro_alpha_s = torch.nn.Parameter(
            torch.randn(n_input_shared,device='cuda')
        )
        self.background_pro_log_beta_s = torch.nn.Parameter(
            torch.clamp(torch.randn(n_input_shared,device='cuda'), -10, 1)
        )

        self.background_pro_alpha_us = torch.nn.Parameter(
            torch.randn(n_input_protein_unshared,device='cuda')
        )
        self.background_pro_log_beta_us = torch.nn.Parameter(
            torch.clamp(torch.randn(n_input_protein_unshared,device='cuda'), -10, 1)
        )

        self.px_r_us = torch.exp(torch.nn.Parameter(torch.randn(n_input_rna_unshared,device='cuda')))
        self.py_r_us = torch.exp(torch.nn.Parameter(torch.ones(n_input_protein_unshared,device='cuda')))
        self.px_r_s = torch.exp(torch.nn.Parameter(torch.randn(n_input_shared,device='cuda')))
        self.py_r_s = torch.exp(torch.nn.Parameter(torch.ones(n_input_shared,device='cuda')))
        
        ## Encoders
        self.shared_encoder = EncoderTOTALVI(
            n_input_shared+1,
            n_latent,
            n_layers=n_layers_encoder,
            n_cat_list=[n_batch] if encoder_batch else None,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_encoder,
            distribution=latent_distribution,
        )
        self.rna_unshared_encoder = EncoderTOTALVI(
            n_input_rna_unshared+1,
            n_latent,
            n_layers=n_layers_encoder,
            n_cat_list=[n_batch] if encoder_batch else None,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_encoder,
            distribution=latent_distribution,
        )
        self.protein_unshared_encoder = EncoderProteinVI(
            n_input_protein_unshared+1,
            n_latent,
            n_layers=n_layers_encoder,
            n_cat_list=[n_batch] if encoder_batch else None,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_encoder,
            distribution=latent_distribution,
        )

        ## Product of Experts
        self.experts = ProductOfExperts()

        ### Decoder Shared
        self.decoder_share = build_multi_layers([n_latent+1,n_hidden//2,n_hidden], dropout_rate = dropout_rate_encoder)

        ### Decoders
        # self.rna_decoder = DecoderSCVI(
        #     n_input =  n_hidden,
        #     n_output = n_input_rna_unshared,
        #     n_layers = n_layers_decoder,
        #     n_cat_list=[n_batch],
        #     n_hidden=n_hidden,
        #     dropout_rate=dropout_rate_decoder,
        # )
        self.protein_decoder = DecoderproteinVI(
            n_input =  n_hidden,
            n_output_proteins = n_input_protein_unshared,
            n_layers = n_layers_decoder,
            n_cat_list=[n_batch],
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_decoder,
        )

        self.rna_decoder_unshared = DecoderTOTALVIRNA(
            n_hidden,
            n_input_rna_unshared,
            n_layers=n_layers_decoder,
            n_cat_list=[n_batch],
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_decoder,
        )

        self.shared_decoder = DecoderTOTALVI_shared(
            n_hidden,
            n_input_shared,
            n_input_shared,
            n_layers=n_layers_decoder,
            n_cat_list=[n_batch],
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_decoder,
        )

        
        py_back_alpha_prior = self.background_pro_alpha_s
        py_back_beta_prior = torch.exp(self.background_pro_log_beta_s)
        self.back_mean_prior_s = Normal(py_back_alpha_prior, py_back_beta_prior)

        py_back_alpha_prior = self.background_pro_alpha_us
        py_back_beta_prior = torch.exp(self.background_pro_log_beta_us)
        self.back_mean_prior_us = Normal(py_back_alpha_prior, py_back_beta_prior)
        
        

    
    def get_reconstruction_loss_rna(
        self, x, px_rate, px_r, px_dropout, **kwargs
    ) -> torch.Tensor:
        x_ = x
        if self.reconstruction_loss_gene == "zinb":
            reconst_loss = (
                -ZeroInflatedNegativeBinomial(
                    mu = px_rate, theta = px_r, zi_logits = px_dropout
                )
                .log_prob(x_)
                .sum(dim=-1)
            )
        else:
            reconst_loss = (
                -NegativeBinomial(mu = px_rate, theta = px_r).log_prob(x_).sum(dim=-1)
            )
        return reconst_loss

    def get_reconstruction_loss_protein(
        self,
        y: torch.Tensor,
        py_: Dict[str, torch.Tensor],
        pro_batch_mask_minibatch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        y_ = y
        reconst_loss_protein_full = -log_mixture_nb(
            y_, py_["rate_back"], py_["rate_fore"], py_["r"], None, py_["mixing"]
        )
        if pro_batch_mask_minibatch is not None:
            temp_pro_loss_full = torch.zeros_like(reconst_loss_protein_full)
            temp_pro_loss_full.masked_scatter_(
                pro_batch_mask_minibatch.bool(), reconst_loss_protein_full
            )

            reconst_loss_protein = temp_pro_loss_full.sum(dim=-1)
        else:
            reconst_loss_protein = reconst_loss_protein_full.sum(dim=-1)

        return reconst_loss_protein     

    def get_reconstruction_loss_shared(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        px_: Dict[str, torch.Tensor],
        py_: Dict[str, torch.Tensor],
        pro_batch_mask_minibatch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute reconstruction loss
        """
        # Reconstruction Loss
        if self.reconstruction_loss_gene == "zinb":
            reconst_loss_gene = (
                -ZeroInflatedNegativeBinomial(
                    mu=px_["rate"], theta=px_["r"], zi_logits=px_["dropout"]
                )
                .log_prob(x)
                .sum(dim=-1)
            )
        else:
            reconst_loss_gene = (
                -NegativeBinomial(mu=px_["rate"], theta=px_["r"])
                .log_prob(x)
                .sum(dim=-1)
            )

        reconst_loss_protein_full = -log_mixture_nb(
            y, py_["rate_back"], py_["rate_fore"], py_["r"], None, py_["mixing"]
        )
        if pro_batch_mask_minibatch is not None:
            temp_pro_loss_full = torch.zeros_like(reconst_loss_protein_full)
            temp_pro_loss_full.masked_scatter_(
                pro_batch_mask_minibatch.bool(), reconst_loss_protein_full
            )

            reconst_loss_protein = temp_pro_loss_full.sum(dim=-1)
        else:
            reconst_loss_protein = reconst_loss_protein_full.sum(dim=-1)

        return reconst_loss_gene, reconst_loss_protein

    def reparameterize_transformation(self, mu, var):
        untran_z = Normal(mu, var.sqrt()).rsample()
        z = untran_z
        return z, untran_z

    def create_dict(self,x_dict,y_dict,ql_msh, ql_vsh):
        log_library_gene = torch.clamp(reparameterize_gaussian(ql_msh, ql_vsh), max=15)
        library_gene = torch.exp(log_library_gene)
        latent = x_dict
        untran_latent = y_dict
        latent["l"] = library_gene
        untran_latent["l"] = log_library_gene
        return latent,untran_latent

    def modify_z_batch(self,gene_size,z):
        z_gene = z[:gene_size]
        z_protein = z[gene_size:]       
        additional_columns = torch.tensor([[0.0]] * z_gene.shape[0], dtype=torch.float32,device='cuda')
        z_gene = torch.cat([z_gene, additional_columns], dim=1)
        additional_columns = torch.tensor([[1.0]] * z_protein.shape[0], dtype=torch.float32,device='cuda')
        z_protein = torch.cat([z_protein, additional_columns], dim=1)
        z = torch.cat((z_gene, z_protein), dim=0)
        return z

    def inference(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_unshared: torch.Tensor,
        y_unshared: torch.Tensor,
        batch_index: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
        n_samples=1,
        transform_batch: Optional[int] = None,
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """ Internal helper function to compute necessary inference quantities

         We use the dictionary ``px_`` to contain the parameters of the ZINB/NB for genes.
         The rate refers to the mean of the NB, dropout refers to Bernoulli mixing parameters.
         `scale` refers to the quanity upon which differential expression is performed. For genes,
         this can be viewed as the mean of the underlying gamma distribution.

         We use the dictionary ``py_`` to contain the parameters of the Mixture NB distribution for proteins.
         `rate_fore` refers to foreground mean, while `rate_back` refers to background mean. ``scale`` refers to
         foreground mean adjusted for background probability and scaled to reside in simplex.
         ``back_alpha`` and ``back_beta`` are the posterior parameters for ``rate_back``.  ``fore_scale`` is the scaling
         factor that enforces `rate_fore` > `rate_back`.

         ``px_["r"]`` and ``py_["r"]`` are the inverse dispersion parameters for genes and protein, respectively.
        """
        x_ = x
        y_ = y
        x_unshared_ = x_unshared
        y_unshared_ = y_unshared
 
        if self.log_variational:
            x_ = torch.log(1.0 + x_)
            y_ = torch.log(1.0 + y_)
            x_unshared_ = torch.log(1.0 + x_unshared_)
            y_unshared_ = torch.log(1.0 + y_unshared_)
        
        additional_columns = torch.tensor([[0.0]] * x_.shape[0], dtype=torch.float32,device='cuda')
        x_unshared_extended = torch.cat([x_unshared_, additional_columns], dim=1)
        additional_columns = torch.tensor([[1.0]] * y_.shape[0], dtype=torch.float32,device='cuda')
        x_unshared_extended_zeroes = torch.cat([torch.zeros((y_.shape[0],x_unshared_.shape[1]),dtype=torch.float32,device='cuda'),additional_columns],dim=1)
        orig_tensor_x_unshared = torch.cat((x_unshared_extended, x_unshared_extended_zeroes), dim=0)

        additional_columns = torch.tensor([[0.0]] * x_.shape[0], dtype=torch.float32,device='cuda')
        y_unshared_extended_zeroes = torch.cat([torch.zeros((x_.shape[0],y_unshared_.shape[1]),dtype=torch.float32,device='cuda'),additional_columns],dim=1)
        additional_columns = torch.tensor([[1.0]] * y_.shape[0], dtype=torch.float32,device='cuda')
        y_unshared_extended = torch.cat([y_unshared_, additional_columns], dim=1)
        orig_tensor_y_unshared = torch.cat((y_unshared_extended_zeroes,y_unshared_extended), dim=0)
        x_size = x.shape[0]

        additional_columns = torch.tensor([[0.0]] * x_.shape[0], dtype=torch.float32,device='cuda')
        x_extended = torch.cat([x_, additional_columns], dim=1)
        additional_columns = torch.tensor([[1.0]] * y_.shape[0], dtype=torch.float32,device='cuda')
        y_extended = torch.cat([y_, additional_columns], dim=1)
        orig_tensor = torch.cat((x_extended, y_extended), dim=0)


        qz_mruns, qz_vruns,ql_mruns, ql_vruns,latentrunsh, untran_latentrunsh = self.rna_unshared_encoder(orig_tensor_x_unshared,batch_index)
        qz_msh, qz_vsh, ql_msh, ql_vsh, latentsh, untran_latentsh = self.shared_encoder(orig_tensor, batch_index)
        qz_mpunsh, qz_vpunsh, f_, g_ = self.protein_unshared_encoder(orig_tensor_y_unshared, batch_index)


        batch_size_ = (1, qz_msh.shape[0], qz_msh.shape[1])
        
        means   = Variable(torch.zeros(batch_size_,device='cuda'))
        var = Variable(torch.ones(batch_size_,device='cuda'))
        means = torch.cat((means,qz_mruns.unsqueeze(0)), dim=0)
        var  = torch.cat((var,qz_vruns.unsqueeze(0)), dim=0)
        means = torch.cat((means,qz_msh.unsqueeze(0)), dim=0)
        var  = torch.cat((var,qz_vsh.unsqueeze(0)), dim=0)
        means = torch.cat((means,qz_mpunsh.unsqueeze(0)), dim=0)
        var  = torch.cat((var,qz_vpunsh.unsqueeze(0)), dim=0)

		# Combine the gaussians
        qz_m, qz_v = self.experts(means, var)
        z, untran_z = self.reparameterize_transformation(qz_m, qz_v)
        gene_size = x_.shape[0]
        z = self.modify_z_batch(gene_size,z)
        latent = {}
        latent["z"] = z
        untran_latent = {}
        untran_latent["z"] = z
        z = latent["z"]
        untran_z = untran_latent["z"]
        k = self.decoder_share(z)

        ### Shared part
        library_gene = latentsh["l"]
        untran_l = untran_latentsh["l"]
        px_, py_, log_pro_back_mean = self.shared_decoder(x_size,k,library_gene,batch_index,label)
        px_["r"] = self.px_r_s
        py_["r"] = self.py_r_s
        shared_dict = dict(
            px_=px_,
            py_=py_,
            qz_m=qz_m,
            qz_v=qz_v,
            z=z,
            untran_z=untran_z,
            ql_m=ql_msh,
            ql_v=ql_vsh,
            library_gene=library_gene,
            untran_l=untran_l,
            log_pro_back_mean=log_pro_back_mean,
        )

        ## Unshared RNA
        library_gene = latentrunsh["l"]
        untran_l = untran_latentrunsh["l"]
        
        px_ = self.rna_decoder_unshared(x_size,k,library_gene,batch_index,label)
        unshared_rna_dict = dict(
            px_scale=px_["scale"],
            px_r=self.px_r_us,
            px_rate=px_["rate"],
            px_dropout=px_["dropout"],
            qz_m=qz_m,
            qz_v=qz_v,
            z=z,
            ql_m=ql_mruns,
            ql_v=ql_vruns,
            library=library_gene,
        )

        ### Unshared protein
        py_, log_pro_back_mean = self.protein_decoder(k[gene_size:],None)
        py_["r"] = self.py_r_us
        unshared_protein_dict =  dict(
            py_= py_,
            qz_m = qz_m,
            qz_v = qz_v,
            z=z,
            untran_z=untran_z,
            log_pro_back_mean=log_pro_back_mean,
        )

        return shared_dict,unshared_rna_dict,unshared_protein_dict

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_unshared: torch.Tensor,
        y_unshared: torch.Tensor,
        local_l_mean_gene: torch.Tensor,
        local_l_var_gene: torch.Tensor,
        batch_index: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor
    ]:
        """ Returns the reconstruction loss and the Kullback divergences

        :param x: tensor of values with shape ``(batch_size, n_input_genes)``
        :param y: tensor of values with shape ``(batch_size, n_input_proteins)``
        :param local_l_mean_gene: tensor of means of the prior distribution of latent variable l
         with shape ``(batch_size, 1)````
        :param local_l_var_gene: tensor of variancess of the prior distribution of latent variable l
         with shape ``(batch_size, 1)``
        :param batch_index: array that indicates which batch the cells belong to with shape ``batch_size``
        :param label: tensor of cell-types labels with shape (batch_size, n_labels)
        :return: the reconstruction loss and the Kullback divergences
        """
        # Parameters for z latent distribution
        shared_dict,unshared_rna_dict,unshared_protein_dict = self.inference(x,y,x_unshared,y_unshared,batch_index,label)
    
        ## Globally lets talk bullshit
        qz_m = shared_dict["qz_m"]
        qz_v = shared_dict["qz_v"]
        z = shared_dict["z"]
        kl_div_z = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(0, 1)).sum(dim=1)
        z_gene = z[:x.shape[0]]
        z_protein = z[x.shape[0]:]

        ## Lets take shared features
        ql_m_shared = shared_dict["ql_m"]
        ql_v_shared = shared_dict["ql_v"]
        px_shared = shared_dict["px_"]
        py_shared = shared_dict["py_"]
        reconst_loss_gene_shared, reconst_loss_protein_shared = self.get_reconstruction_loss_shared(
            x, y, px_shared, py_shared, None
        )        
        kl_div_l_gene_shared = kl(
            Normal(ql_m_shared[:x.shape[0]], torch.sqrt(ql_v_shared[:x.shape[0]])),
            Normal(local_l_mean_gene, torch.sqrt(local_l_var_gene)),
        ).sum(dim=1)

        kl_div_back_pro_shared = kl(Normal(py_shared["back_alpha"], py_shared["back_beta"]), self.back_mean_prior_s).sum(dim=1)
        ## Lets take unshared features of RNA
        ql_m_unshared = unshared_rna_dict["ql_m"]
        ql_v_unshared = unshared_rna_dict["ql_v"]
        px_rate = unshared_rna_dict["px_rate"]
        px_r = unshared_rna_dict["px_r"]
        px_dropout = unshared_rna_dict["px_dropout"]
        reconst_loss_gene_unshared = self.get_reconstruction_loss_rna(x_unshared, px_rate, px_r, px_dropout)
        kl_div_l_gene_unshared = kl(
            Normal(ql_m_unshared[:x.shape[0]], torch.sqrt(ql_v_unshared[:x.shape[0]])),
            Normal(local_l_mean_gene, torch.sqrt(local_l_var_gene)),
        ).sum(dim=1)
        ## Lets take unshared features of Protein
        py_ = unshared_protein_dict["py_"]
        reconst_loss_protein_unshared = self.get_reconstruction_loss_protein(
            y_unshared, py_, None
        )
        kl_div_back_pro_unshared = kl(
            Normal(py_["back_alpha"], py_["back_beta"]), self.back_mean_prior_us
        ).sum(dim=1)

                
        return (
            reconst_loss_gene_unshared,
            reconst_loss_gene_shared,
            kl_div_l_gene_unshared,
            kl_div_l_gene_shared,
            reconst_loss_protein_unshared,
            reconst_loss_protein_shared,
            kl_div_back_pro_unshared,
            kl_div_back_pro_shared,
            kl_div_z,
            z_gene,
            z_protein,
        )

class TOTALVI_shared_sirNN(nn.Module):
    """Total variational inference for CITE-seq data
    Implements the totalVI model of [Gayoso19]_.
    :param n_input_genes: Number of input genes
    :param n_input_proteins: Number of input proteins
    :param n_batch: Number of batches
    :param n_labels: Number of labels
    :param n_hidden: Number of nodes per hidden layer for the z encoder (protein+genes),
                     genes library encoder, z->genes+proteins decoder
    :param n_latent: Dimensionality of the latent space
    :param n_layers: Number of hidden layers used for encoder and decoder NNs
    :param dropout_rate: Dropout rate for neural networks
    :param genes_dispersion: One of the following

        * ``'gene'`` - genes_dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - genes_dispersion can differ between different batches
        * ``'gene-label'`` - genes_dispersion can differ between different labels

    :param protein_dispersion: One of the following

        * ``'protein'`` - protein_dispersion parameter is constant per protein across cells
        * ``'protein-batch'`` - protein_dispersion can differ between different batches NOT TESTED
        * ``'protein-label'`` - protein_dispersion can differ between different labels NOT TESTED

    :param log_variational: Log(data+1) prior to encoding for numerical stability. Not normalization.
    :param reconstruction_loss_genes:  One of

        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution

    :param latent_distribution:  One of

        * ``'normal'`` - Isotropic normal
        * ``'ln'`` - Logistic normal with normal params N(0, 1)

    Examples:
        >>> dataset = Dataset10X(dataset_name="pbmc_10k_protein_v3", save_path=save_path)
        >>> totalvae = TOTALVI(gene_dataset.nb_genes, len(dataset.protein_names), use_cuda=True)
    """

    def __init__(
        self,
        n_input_shared: int,
        n_input_rna_unshared: int,
        n_input_protein_unshared: int,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 256,
        n_latent: int = 20,
        n_layers_encoder: int = 1,
        n_layers_decoder: int = 1,
        dropout_rate_decoder: float = 0.1,
        dropout_rate_encoder: float = 0.1,
        gene_dispersion: str = "gene",
        protein_dispersion: str = "protein",
        log_variational: bool = True,
        reconstruction_loss_gene: str = "nb",
        latent_distribution: str = "normal",
        protein_batch_mask: List[np.ndarray] = None,
        encoder_batch: bool = True,
    ):
        super().__init__()
        self.gene_dispersion = gene_dispersion
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.reconstruction_loss_gene = reconstruction_loss_gene
        self.n_batch = n_batch
        self.n_labels = n_labels
        self.protein_dispersion = protein_dispersion
        self.latent_distribution = latent_distribution
        self.protein_batch_mask = protein_batch_mask
        self.background_pro_alpha_s = torch.nn.Parameter(
            torch.randn(n_input_shared,device='cuda')
        )
        self.background_pro_log_beta_s = torch.nn.Parameter(
            torch.clamp(torch.randn(n_input_shared,device='cuda'), -10, 1)
        )

        self.background_pro_alpha_us = torch.nn.Parameter(
            torch.randn(n_input_protein_unshared,device='cuda')
        )
        self.background_pro_log_beta_us = torch.nn.Parameter(
            torch.clamp(torch.randn(n_input_protein_unshared,device='cuda'), -10, 1)
        )
        self.px_r_us = torch.exp(torch.nn.Parameter(torch.randn(n_input_rna_unshared,device='cuda')))
        self.py_r_us = torch.exp(torch.nn.Parameter(torch.ones(n_input_protein_unshared,device='cuda')))
        self.px_r_s = torch.exp(torch.nn.Parameter(torch.randn(n_input_shared,device='cuda')))
        self.py_r_s = torch.exp(torch.nn.Parameter(torch.ones(n_input_shared,device='cuda')))

        py_back_alpha_prior = self.background_pro_alpha_s
        py_back_beta_prior = torch.exp(self.background_pro_log_beta_s)
        self.back_mean_prior_s = Normal(py_back_alpha_prior, py_back_beta_prior)

        py_back_alpha_prior = self.background_pro_alpha_us
        py_back_beta_prior = torch.exp(self.background_pro_log_beta_us)
        self.back_mean_prior_us = Normal(py_back_alpha_prior, py_back_beta_prior)
        
        ## Encoders
        self.shared_encoder = EncoderTOTALVI(
            n_input_shared+1,
            n_latent,
            n_layers=n_layers_encoder,
            n_cat_list=[n_batch] if encoder_batch else None,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_encoder,
            distribution=latent_distribution,
        )
        self.rna_unshared_encoder = EncoderTOTALVI(
            n_input_rna_unshared+1,
            n_latent,
            n_layers=n_layers_encoder,
            n_cat_list=[n_batch] if encoder_batch else None,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_encoder,
            distribution=latent_distribution,
        )
        self.protein_unshared_encoder = EncoderProteinVI(
            n_input_protein_unshared+1,
            n_latent,
            n_layers=n_layers_encoder,
            n_cat_list=[n_batch] if encoder_batch else None,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_encoder,
            distribution=latent_distribution,
        )

        ### Encoder Shared
        self.encoder_share = build_multi_layers([(n_latent)*4,n_latent,n_hidden], dropout_rate = dropout_rate_decoder)
        ### Decoder Shared
        self.decoder_share = build_multi_layers([n_latent+1,n_hidden//2,n_hidden], dropout_rate = dropout_rate_decoder)

        
        self.protein_decoder = DecoderproteinVI(
            n_input =  n_hidden,
            n_output_proteins = n_input_protein_unshared,
            n_layers = n_layers_decoder,
            n_cat_list=[n_batch],
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_decoder,
        )

        self.rna_decoder_unshared = DecoderTOTALVIRNA(
            n_hidden,
            n_input_rna_unshared,
            n_layers=n_layers_decoder,
            n_cat_list=[n_batch],
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_decoder,
        )

        self.shared_decoder = DecoderTOTALVI_shared(
            n_hidden,
            n_input_shared,
            n_input_shared,
            n_layers=n_layers_decoder,
            n_cat_list=[n_batch],
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_decoder,
        )
        
        
        

    
    def get_reconstruction_loss_rna(
        self, x, px_rate, px_r, px_dropout, **kwargs
    ) -> torch.Tensor:
        x_ = x
        if self.reconstruction_loss_gene == "zinb":
            reconst_loss = (
                -ZeroInflatedNegativeBinomial(
                    mu = px_rate, theta = px_r, zi_logits = px_dropout
                )
                .log_prob(x_)
                .sum(dim=-1)
            )
        else:
            reconst_loss = (
                -NegativeBinomial(mu = px_rate, theta = px_r).log_prob(x_).sum(dim=-1)
            )
        return reconst_loss

    def get_reconstruction_loss_protein(
        self,
        y: torch.Tensor,
        py_: Dict[str, torch.Tensor],
        pro_batch_mask_minibatch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        y_ = y
        reconst_loss_protein_full = -log_mixture_nb(
            y_, py_["rate_back"], py_["rate_fore"], py_["r"], None, py_["mixing"]
        )
        if pro_batch_mask_minibatch is not None:
            temp_pro_loss_full = torch.zeros_like(reconst_loss_protein_full)
            temp_pro_loss_full.masked_scatter_(
                pro_batch_mask_minibatch.bool(), reconst_loss_protein_full
            )

            reconst_loss_protein = temp_pro_loss_full.sum(dim=-1)
        else:
            reconst_loss_protein = reconst_loss_protein_full.sum(dim=-1)

        return reconst_loss_protein     

    def get_reconstruction_loss_shared(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        px_: Dict[str, torch.Tensor],
        py_: Dict[str, torch.Tensor],
        pro_batch_mask_minibatch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute reconstruction loss
        """
        # Reconstruction Loss
        if self.reconstruction_loss_gene == "zinb":
            reconst_loss_gene = (
                -ZeroInflatedNegativeBinomial(
                    mu=px_["rate"], theta=px_["r"], zi_logits=px_["dropout"]
                )
                .log_prob(x)
                .sum(dim=-1)
            )
        else:
            reconst_loss_gene = (
                -NegativeBinomial(mu=px_["rate"], theta=px_["r"])
                .log_prob(x)
                .sum(dim=-1)
            )

        reconst_loss_protein_full = -log_mixture_nb(
            y, py_["rate_back"], py_["rate_fore"], py_["r"], None, py_["mixing"]
        )
        if pro_batch_mask_minibatch is not None:
            temp_pro_loss_full = torch.zeros_like(reconst_loss_protein_full)
            temp_pro_loss_full.masked_scatter_(
                pro_batch_mask_minibatch.bool(), reconst_loss_protein_full
            )

            reconst_loss_protein = temp_pro_loss_full.sum(dim=-1)
        else:
            reconst_loss_protein = reconst_loss_protein_full.sum(dim=-1)

        return reconst_loss_gene, reconst_loss_protein

    def reparameterize_transformation(self, mu, var):
        untran_z = Normal(mu, var.sqrt()).rsample()
        z = untran_z
        return z, untran_z

    def create_dict(self,x_dict,y_dict,ql_msh, ql_vsh):
        log_library_gene = torch.clamp(reparameterize_gaussian(ql_msh, ql_vsh), max=15)
        library_gene = torch.exp(log_library_gene)
        latent = x_dict
        untran_latent = y_dict
        latent["l"] = library_gene
        untran_latent["l"] = log_library_gene
        return latent,untran_latent

    def modify_z_batch(self,gene_size,z):
        z_gene = z[:gene_size]
        z_protein = z[gene_size:]       
        additional_columns = torch.tensor([[0.0]] * z_gene.shape[0], dtype=torch.float32,device='cuda')
        z_gene = torch.cat([z_gene, additional_columns], dim=1)
        additional_columns = torch.tensor([[1.0]] * z_protein.shape[0], dtype=torch.float32,device='cuda')
        z_protein = torch.cat([z_protein, additional_columns], dim=1)
        z = torch.cat((z_gene, z_protein), dim=0)
        return z

    def inference(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_unshared: torch.Tensor,
        y_unshared: torch.Tensor,
        batch_index: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
        n_samples=1,
        transform_batch: Optional[int] = None,
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """ Internal helper function to compute necessary inference quantities

         We use the dictionary ``px_`` to contain the parameters of the ZINB/NB for genes.
         The rate refers to the mean of the NB, dropout refers to Bernoulli mixing parameters.
         `scale` refers to the quanity upon which differential expression is performed. For genes,
         this can be viewed as the mean of the underlying gamma distribution.

         We use the dictionary ``py_`` to contain the parameters of the Mixture NB distribution for proteins.
         `rate_fore` refers to foreground mean, while `rate_back` refers to background mean. ``scale`` refers to
         foreground mean adjusted for background probability and scaled to reside in simplex.
         ``back_alpha`` and ``back_beta`` are the posterior parameters for ``rate_back``.  ``fore_scale`` is the scaling
         factor that enforces `rate_fore` > `rate_back`.

         ``px_["r"]`` and ``py_["r"]`` are the inverse dispersion parameters for genes and protein, respectively.
        """
        x_ = x
        y_ = y
        x_unshared_ = x_unshared
        y_unshared_ = y_unshared
 
        if self.log_variational:
            x_ = torch.log(1.0 + x_)
            y_ = torch.log(1.0 + y_)
            x_unshared_ = torch.log(1.0 + x_unshared_)
            y_unshared_ = torch.log(1.0 + y_unshared_)
        
        additional_columns = torch.tensor([[0.0]] * x_.shape[0], dtype=torch.float32,device='cuda')
        x_unshared_extended = torch.cat([x_unshared_, additional_columns], dim=1)
        additional_columns = torch.tensor([[1.0]] * y_.shape[0], dtype=torch.float32,device='cuda')
        x_unshared_extended_zeroes = torch.cat([torch.zeros((y_.shape[0],x_unshared_.shape[1]),dtype=torch.float32,device='cuda'),additional_columns],dim=1)
        orig_tensor_x_unshared = torch.cat((x_unshared_extended, x_unshared_extended_zeroes), dim=0)

        additional_columns = torch.tensor([[0.0]] * x_.shape[0], dtype=torch.float32,device='cuda')
        y_unshared_extended_zeroes = torch.cat([torch.zeros((x_.shape[0],y_unshared_.shape[1]),dtype=torch.float32,device='cuda'),additional_columns],dim=1)
        additional_columns = torch.tensor([[1.0]] * y_.shape[0], dtype=torch.float32,device='cuda')
        y_unshared_extended = torch.cat([y_unshared_, additional_columns], dim=1)
        orig_tensor_y_unshared = torch.cat((y_unshared_extended_zeroes,y_unshared_extended), dim=0)
        x_size = x.shape[0]

        additional_columns = torch.tensor([[0.0]] * x_.shape[0], dtype=torch.float32,device='cuda')
        x_extended = torch.cat([x_, additional_columns], dim=1)
        additional_columns = torch.tensor([[1.0]] * y_.shape[0], dtype=torch.float32,device='cuda')
        y_extended = torch.cat([y_, additional_columns], dim=1)
        orig_tensor = torch.cat((x_extended, y_extended), dim=0)


        qz_mruns, qz_vruns,ql_mruns, ql_vruns,latentrunsh, untran_latentrunsh = self.rna_unshared_encoder(orig_tensor_x_unshared,batch_index)
        qz_msh, qz_vsh, ql_msh, ql_vsh, latentsh, untran_latentsh = self.shared_encoder(orig_tensor, batch_index)
        qz_mpunsh, qz_vpunsh, f_, g_ = self.protein_unshared_encoder(orig_tensor_y_unshared, batch_index)


        # batch_size_ = (1, qz_msh.shape[0], qz_msh.shape[1])
        
        # means   = Variable(torch.zeros(batch_size_,device='cuda'))
        # var = Variable(torch.ones(batch_size_,device='cuda'))
        # means = torch.cat((means,qz_mruns.unsqueeze(0)), dim=0)
        # var  = torch.cat((var,qz_vruns.unsqueeze(0)), dim=0)
        # means = torch.cat((means,qz_msh.unsqueeze(0)), dim=0)
        # var  = torch.cat((var,qz_vsh.unsqueeze(0)), dim=0)
        # means = torch.cat((means,qz_mpunsh.unsqueeze(0)), dim=0)
        # var  = torch.cat((var,qz_vpunsh.unsqueeze(0)), dim=0)

		# # Combine the gaussians
        # qz_m, qz_v = self.experts(means, var)
        # z, untran_z = self.reparameterize_transformation(qz_m, qz_v)

        z = torch.concat((latentrunsh["z"],latentsh["z"],f_["z"]),dim=1)
        gene_size = x_.shape[0]
        z = self.modify_z_batch(gene_size,z)
        latent = {}
        latent["z"] = z
        untran_latent = {}
        untran_latent["z"] = z
        z = latent["z"]
        untran_z = untran_latent["z"]
        k = self.decoder_share(z)

        ### Shared part
        library_gene = latentsh["l"]
        untran_l = untran_latentsh["l"]
        px_, py_, log_pro_back_mean = self.shared_decoder(x_size,k,library_gene,batch_index,label)
        px_["r"] = self.px_r_s
        py_["r"] = self.py_r_s
        shared_dict = dict(
            px_=px_,
            py_=py_,
            qz_m=qz_m,
            qz_v=qz_v,
            z=z,
            untran_z=untran_z,
            ql_m=ql_msh,
            ql_v=ql_vsh,
            library_gene=library_gene,
            untran_l=untran_l,
            log_pro_back_mean=log_pro_back_mean,
        )

        ## Unshared RNA
        library_gene = latentrunsh["l"]
        untran_l = untran_latentrunsh["l"]
        
        px_ = self.rna_decoder_unshared(x_size,k,library_gene,batch_index,label)
        unshared_rna_dict = dict(
            px_scale=px_["scale"],
            px_r=self.px_r_us,
            px_rate=px_["rate"],
            px_dropout=px_["dropout"],
            qz_m=qz_m,
            qz_v=qz_v,
            z=z,
            ql_m=ql_mruns,
            ql_v=ql_vruns,
            library=library_gene,
        )

        ### Unshared protein
        py_, log_pro_back_mean = self.protein_decoder(k[gene_size:],None)
        py_["r"] = self.py_r_us
        unshared_protein_dict =  dict(
            py_= py_,
            qz_m = qz_m,
            qz_v = qz_v,
            z=z,
            untran_z=untran_z,
            log_pro_back_mean=log_pro_back_mean,
        )

        return shared_dict,unshared_rna_dict,unshared_protein_dict

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_unshared: torch.Tensor,
        y_unshared: torch.Tensor,
        local_l_mean_gene: torch.Tensor,
        local_l_var_gene: torch.Tensor,
        batch_index: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor
    ]:
        """ Returns the reconstruction loss and the Kullback divergences

        :param x: tensor of values with shape ``(batch_size, n_input_genes)``
        :param y: tensor of values with shape ``(batch_size, n_input_proteins)``
        :param local_l_mean_gene: tensor of means of the prior distribution of latent variable l
         with shape ``(batch_size, 1)````
        :param local_l_var_gene: tensor of variancess of the prior distribution of latent variable l
         with shape ``(batch_size, 1)``
        :param batch_index: array that indicates which batch the cells belong to with shape ``batch_size``
        :param label: tensor of cell-types labels with shape (batch_size, n_labels)
        :return: the reconstruction loss and the Kullback divergences
        """
        # Parameters for z latent distribution
        shared_dict,unshared_rna_dict,unshared_protein_dict = self.inference(x,y,x_unshared,y_unshared,batch_index,label)
    
        ## Globally lets talk bullshit
        qz_m = shared_dict["qz_m"]
        qz_v = shared_dict["qz_v"]
        z = shared_dict["z"]
        kl_div_z = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(0, 1)).sum(dim=1)
        z_gene = z[:x.shape[0]]
        z_protein = z[x.shape[0]:]

        ## Lets take shared features
        ql_m_shared = shared_dict["ql_m"]
        ql_v_shared = shared_dict["ql_v"]
        px_shared = shared_dict["px_"]
        py_shared = shared_dict["py_"]
        reconst_loss_gene_shared, reconst_loss_protein_shared = self.get_reconstruction_loss_shared(
            x, y, px_shared, py_shared, None
        )        
        kl_div_l_gene_shared = kl(
            Normal(ql_m_shared[:x.shape[0]], torch.sqrt(ql_v_shared[:x.shape[0]])),
            Normal(local_l_mean_gene, torch.sqrt(local_l_var_gene)),
        ).sum(dim=1)

        kl_div_back_pro_shared = kl(Normal(py_shared["back_alpha"], py_shared["back_beta"]), self.back_mean_prior_s).sum(dim=1)
        ## Lets take unshared features of RNA
        ql_m_unshared = unshared_rna_dict["ql_m"]
        ql_v_unshared = unshared_rna_dict["ql_v"]
        px_rate = unshared_rna_dict["px_rate"]
        px_r = unshared_rna_dict["px_r"]
        px_dropout = unshared_rna_dict["px_dropout"]
        reconst_loss_gene_unshared = self.get_reconstruction_loss_rna(x_unshared, px_rate, px_r, px_dropout)
        kl_div_l_gene_unshared = kl(
            Normal(ql_m_unshared[:x.shape[0]], torch.sqrt(ql_v_unshared[:x.shape[0]])),
            Normal(local_l_mean_gene, torch.sqrt(local_l_var_gene)),
        ).sum(dim=1)
        ## Lets take unshared features of Protein
        py_ = unshared_protein_dict["py_"]
        reconst_loss_protein_unshared = self.get_reconstruction_loss_protein(
            y_unshared, py_, None
        )
        kl_div_back_pro_unshared = kl(
            Normal(py_["back_alpha"], py_["back_beta"]), self.back_mean_prior_us
        ).sum(dim=1)

                
        return (
            reconst_loss_gene_unshared,
            reconst_loss_gene_shared,
            kl_div_l_gene_unshared,
            kl_div_l_gene_shared,
            reconst_loss_protein_unshared,
            reconst_loss_protein_shared,
            kl_div_back_pro_unshared,
            kl_div_back_pro_shared,
            kl_div_z,
            z_gene,
            z_protein,
        )

class TOTALVI_shared_sirPOE_sir(nn.Module):
    """Total variational inference for CITE-seq data
    Implements the totalVI model of [Gayoso19]_.
    :param n_input_genes: Number of input genes
    :param n_input_proteins: Number of input proteins
    :param n_batch: Number of batches
    :param n_labels: Number of labels
    :param n_hidden: Number of nodes per hidden layer for the z encoder (protein+genes),
                     genes library encoder, z->genes+proteins decoder
    :param n_latent: Dimensionality of the latent space
    :param n_layers: Number of hidden layers used for encoder and decoder NNs
    :param dropout_rate: Dropout rate for neural networks
    :param genes_dispersion: One of the following

        * ``'gene'`` - genes_dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - genes_dispersion can differ between different batches
        * ``'gene-label'`` - genes_dispersion can differ between different labels

    :param protein_dispersion: One of the following

        * ``'protein'`` - protein_dispersion parameter is constant per protein across cells
        * ``'protein-batch'`` - protein_dispersion can differ between different batches NOT TESTED
        * ``'protein-label'`` - protein_dispersion can differ between different labels NOT TESTED

    :param log_variational: Log(data+1) prior to encoding for numerical stability. Not normalization.
    :param reconstruction_loss_genes:  One of

        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution

    :param latent_distribution:  One of

        * ``'normal'`` - Isotropic normal
        * ``'ln'`` - Logistic normal with normal params N(0, 1)

    Examples:
        >>> dataset = Dataset10X(dataset_name="pbmc_10k_protein_v3", save_path=save_path)
        >>> totalvae = TOTALVI(gene_dataset.nb_genes, len(dataset.protein_names), use_cuda=True)
    """

    def __init__(
        self,
        n_input_shared: int,
        n_input_rna_unshared: int,
        n_input_protein_unshared: int,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 256,
        n_latent: int = 20,
        n_layers_encoder: int = 1,
        n_layers_decoder: int = 1,
        dropout_rate_decoder: float = 0.1,
        dropout_rate_encoder: float = 0.1,
        gene_dispersion: str = "gene",
        protein_dispersion: str = "protein",
        log_variational: bool = True,
        reconstruction_loss_gene: str = "nb",
        latent_distribution: str = "normal",
        protein_batch_mask: List[np.ndarray] = None,
        encoder_batch: bool = True,
    ):
        super().__init__()
        self.gene_dispersion = gene_dispersion
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.reconstruction_loss_gene = reconstruction_loss_gene
        self.n_batch = n_batch
        self.n_labels = n_labels
        self.protein_dispersion = protein_dispersion
        self.latent_distribution = latent_distribution
        self.protein_batch_mask = protein_batch_mask
        
        self.background_pro_alpha_s = torch.nn.Parameter(
            torch.randn(n_input_shared,device='cuda')
        )
        self.background_pro_log_beta_s = torch.nn.Parameter(
            torch.clamp(torch.randn(n_input_shared,device='cuda'), -10, 1)
        )

        self.background_pro_alpha_us = torch.nn.Parameter(
            torch.randn(n_input_protein_unshared,device='cuda')
        )
        self.background_pro_log_beta_us = torch.nn.Parameter(
            torch.clamp(torch.randn(n_input_protein_unshared,device='cuda'), -10, 1)
        )

        self.px_r_us = torch.exp(torch.nn.Parameter(torch.randn(n_input_rna_unshared,device='cuda')))
        self.py_r_us = torch.exp(torch.nn.Parameter(torch.ones(n_input_protein_unshared,device='cuda')))
        self.px_r_s = torch.exp(torch.nn.Parameter(torch.randn(n_input_shared,device='cuda')))
        self.py_r_s = torch.exp(torch.nn.Parameter(torch.ones(n_input_shared,device='cuda')))
        
        ## Encoders
        self.shared_encoder = EncoderTOTALVI(
            n_input_shared+1,
            n_latent,
            n_layers=n_layers_encoder,
            n_cat_list=[n_batch] if encoder_batch else None,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_encoder,
            distribution=latent_distribution,
        )
        self.rna_unshared_encoder = EncoderTOTALVI(
            n_input_rna_unshared+1,
            n_latent,
            n_layers=n_layers_encoder,
            n_cat_list=[n_batch] if encoder_batch else None,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_encoder,
            distribution=latent_distribution,
        )
        self.protein_unshared_encoder = EncoderProteinVI(
            n_input_protein_unshared+1,
            n_latent,
            n_layers=n_layers_encoder,
            n_cat_list=[n_batch] if encoder_batch else None,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_encoder,
            distribution=latent_distribution,
        )

        ## Product of Experts
        self.experts = ProductOfExperts()

        ### Decoder Shared
        self.decoder_share = build_multi_layers([n_latent+1,n_hidden//2,n_hidden], dropout_rate = dropout_rate_encoder)

        ### Decoders
        # self.rna_decoder = DecoderSCVI(
        #     n_input =  n_hidden,
        #     n_output = n_input_rna_unshared,
        #     n_layers = n_layers_decoder,
        #     n_cat_list=[n_batch],
        #     n_hidden=n_hidden,
        #     dropout_rate=dropout_rate_decoder,
        # )
        self.protein_decoder = DecoderproteinVI_sir(
            n_input =  n_hidden,
            n_output_proteins = n_input_protein_unshared,
            n_layers = n_layers_decoder,
            n_cat_list=[n_batch],
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_decoder,
        )

        self.rna_decoder_unshared = DecoderTOTALVIRNA_sir(
            n_hidden,
            n_input_rna_unshared,
            n_layers=n_layers_decoder,
            n_cat_list=[n_batch],
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_decoder,
        )

        self.shared_decoder = DecoderTOTALVI_shared_sir(
            n_hidden,
            n_input_shared,
            n_input_shared,
            n_layers=n_layers_decoder,
            n_cat_list=[n_batch],
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_decoder,
        )

        
        py_back_alpha_prior = self.background_pro_alpha_s
        py_back_beta_prior = torch.exp(self.background_pro_log_beta_s)
        self.back_mean_prior_s = Normal(py_back_alpha_prior, py_back_beta_prior)

        py_back_alpha_prior = self.background_pro_alpha_us
        py_back_beta_prior = torch.exp(self.background_pro_log_beta_us)
        self.back_mean_prior_us = Normal(py_back_alpha_prior, py_back_beta_prior)
        
        

    
    def get_reconstruction_loss_rna(
        self, x, px_rate, px_r, px_dropout, **kwargs
    ) -> torch.Tensor:
        x_ = x
        if self.reconstruction_loss_gene == "zinb":
            reconst_loss = (
                -ZeroInflatedNegativeBinomial(
                    mu = px_rate, theta = px_r, zi_logits = px_dropout
                )
                .log_prob(x_)
                .sum(dim=-1)
            )
        else:
            reconst_loss = (
                -NegativeBinomial(mu = px_rate, theta = px_r).log_prob(x_).sum(dim=-1)
            )
        return reconst_loss

    def get_reconstruction_loss_protein(
        self,
        y: torch.Tensor,
        py_: Dict[str, torch.Tensor],
        pro_batch_mask_minibatch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        y_ = y
        reconst_loss_protein_full = -log_mixture_nb(
            y_, py_["rate_back"], py_["rate_fore"], py_["r"], None, py_["mixing"]
        )
        if pro_batch_mask_minibatch is not None:
            temp_pro_loss_full = torch.zeros_like(reconst_loss_protein_full)
            temp_pro_loss_full.masked_scatter_(
                pro_batch_mask_minibatch.bool(), reconst_loss_protein_full
            )

            reconst_loss_protein = temp_pro_loss_full.sum(dim=-1)
        else:
            reconst_loss_protein = reconst_loss_protein_full.sum(dim=-1)

        return reconst_loss_protein     

    def get_reconstruction_loss_shared(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        px_: Dict[str, torch.Tensor],
        py_: Dict[str, torch.Tensor],
        pro_batch_mask_minibatch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute reconstruction loss
        """
        # Reconstruction Loss
        if self.reconstruction_loss_gene == "zinb":
            reconst_loss_gene = (
                -ZeroInflatedNegativeBinomial(
                    mu=px_["rate"], theta=px_["r"], zi_logits=px_["dropout"]
                )
                .log_prob(x)
                .sum(dim=-1)
            )
        else:
            reconst_loss_gene = (
                -NegativeBinomial(mu=px_["rate"], theta=px_["r"])
                .log_prob(x)
                .sum(dim=-1)
            )

        reconst_loss_protein_full = -log_mixture_nb(
            y, py_["rate_back"], py_["rate_fore"], py_["r"], None, py_["mixing"]
        )
        if pro_batch_mask_minibatch is not None:
            temp_pro_loss_full = torch.zeros_like(reconst_loss_protein_full)
            temp_pro_loss_full.masked_scatter_(
                pro_batch_mask_minibatch.bool(), reconst_loss_protein_full
            )

            reconst_loss_protein = temp_pro_loss_full.sum(dim=-1)
        else:
            reconst_loss_protein = reconst_loss_protein_full.sum(dim=-1)

        return reconst_loss_gene, reconst_loss_protein

    def reparameterize_transformation(self, mu, var):
        untran_z = Normal(mu, var.sqrt()).rsample()
        z = untran_z
        return z, untran_z

    def create_dict(self,x_dict,y_dict,ql_msh, ql_vsh):
        log_library_gene = torch.clamp(reparameterize_gaussian(ql_msh, ql_vsh), max=15)
        library_gene = torch.exp(log_library_gene)
        latent = x_dict
        untran_latent = y_dict
        latent["l"] = library_gene
        untran_latent["l"] = log_library_gene
        return latent,untran_latent

    def modify_z_batch(self,gene_size,z):
        z_gene = z[:gene_size]
        z_protein = z[gene_size:]       
        additional_columns = torch.tensor([[0.0]] * z_gene.shape[0], dtype=torch.float32,device='cuda')
        z_gene = torch.cat([z_gene, additional_columns], dim=1)
        additional_columns = torch.tensor([[1.0]] * z_protein.shape[0], dtype=torch.float32,device='cuda')
        z_protein = torch.cat([z_protein, additional_columns], dim=1)
        z = torch.cat((z_gene, z_protein), dim=0)
        return z

    def inference(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_unshared: torch.Tensor,
        y_unshared: torch.Tensor,
        batch_index: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
        n_samples=1,
        transform_batch: Optional[int] = None,
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """ Internal helper function to compute necessary inference quantities

         We use the dictionary ``px_`` to contain the parameters of the ZINB/NB for genes.
         The rate refers to the mean of the NB, dropout refers to Bernoulli mixing parameters.
         `scale` refers to the quanity upon which differential expression is performed. For genes,
         this can be viewed as the mean of the underlying gamma distribution.

         We use the dictionary ``py_`` to contain the parameters of the Mixture NB distribution for proteins.
         `rate_fore` refers to foreground mean, while `rate_back` refers to background mean. ``scale`` refers to
         foreground mean adjusted for background probability and scaled to reside in simplex.
         ``back_alpha`` and ``back_beta`` are the posterior parameters for ``rate_back``.  ``fore_scale`` is the scaling
         factor that enforces `rate_fore` > `rate_back`.

         ``px_["r"]`` and ``py_["r"]`` are the inverse dispersion parameters for genes and protein, respectively.
        """
        x_ = x
        y_ = y
        x_unshared_ = x_unshared
        y_unshared_ = y_unshared
 
        if self.log_variational:
            x_ = torch.log(1.0 + x_)
            y_ = torch.log(1.0 + y_)
            x_unshared_ = torch.log(1.0 + x_unshared_)
            y_unshared_ = torch.log(1.0 + y_unshared_)
        
        additional_columns = torch.tensor([[0.0]] * x_.shape[0], dtype=torch.float32,device='cuda')
        x_unshared_extended = torch.cat([x_unshared_, additional_columns], dim=1)
        additional_columns = torch.tensor([[1.0]] * y_.shape[0], dtype=torch.float32,device='cuda')
        x_unshared_extended_zeroes = torch.cat([torch.zeros((y_.shape[0],x_unshared_.shape[1]),dtype=torch.float32,device='cuda'),additional_columns],dim=1)
        orig_tensor_x_unshared = torch.cat((x_unshared_extended, x_unshared_extended_zeroes), dim=0)

        additional_columns = torch.tensor([[0.0]] * x_.shape[0], dtype=torch.float32,device='cuda')
        y_unshared_extended_zeroes = torch.cat([torch.zeros((x_.shape[0],y_unshared_.shape[1]),dtype=torch.float32,device='cuda'),additional_columns],dim=1)
        additional_columns = torch.tensor([[1.0]] * y_.shape[0], dtype=torch.float32,device='cuda')
        y_unshared_extended = torch.cat([y_unshared_, additional_columns], dim=1)
        orig_tensor_y_unshared = torch.cat((y_unshared_extended_zeroes,y_unshared_extended), dim=0)
        x_size = x.shape[0]

        additional_columns = torch.tensor([[0.0]] * x_.shape[0], dtype=torch.float32,device='cuda')
        x_extended = torch.cat([x_, additional_columns], dim=1)
        additional_columns = torch.tensor([[1.0]] * y_.shape[0], dtype=torch.float32,device='cuda')
        y_extended = torch.cat([y_, additional_columns], dim=1)
        orig_tensor = torch.cat((x_extended, y_extended), dim=0)


        qz_mruns, qz_vruns,ql_mruns, ql_vruns,latentrunsh, untran_latentrunsh = self.rna_unshared_encoder(orig_tensor_x_unshared,batch_index)
        qz_msh, qz_vsh, ql_msh, ql_vsh, latentsh, untran_latentsh = self.shared_encoder(orig_tensor, batch_index)
        qz_mpunsh, qz_vpunsh, f_, g_ = self.protein_unshared_encoder(orig_tensor_y_unshared, batch_index)


        batch_size_ = (1, qz_msh.shape[0], qz_msh.shape[1])
        
        means   = Variable(torch.zeros(batch_size_,device='cuda'))
        var = Variable(torch.ones(batch_size_,device='cuda'))
        means = torch.cat((means,qz_mruns.unsqueeze(0)), dim=0)
        var  = torch.cat((var,qz_vruns.unsqueeze(0)), dim=0)
        means = torch.cat((means,qz_msh.unsqueeze(0)), dim=0)
        var  = torch.cat((var,qz_vsh.unsqueeze(0)), dim=0)
        means = torch.cat((means,qz_mpunsh.unsqueeze(0)), dim=0)
        var  = torch.cat((var,qz_vpunsh.unsqueeze(0)), dim=0)

		# Combine the gaussians
        qz_m, qz_v = self.experts(means, var)
        z, untran_z = self.reparameterize_transformation(qz_m, qz_v)
        gene_size = x_.shape[0]
        z = self.modify_z_batch(gene_size,z)
        latent = {}
        latent["z"] = z
        untran_latent = {}
        untran_latent["z"] = z
        z = latent["z"]
        untran_z = untran_latent["z"]
        k = self.decoder_share(z)

        ### Shared part
        library_gene = latentsh["l"]
        untran_l = untran_latentsh["l"]
        px_, py_, log_pro_back_mean = self.shared_decoder(x_size,k,library_gene,batch_index,label)
        px_["r"] = self.px_r_s
        py_["r"] = self.py_r_s
        shared_dict = dict(
            px_=px_,
            py_=py_,
            qz_m=qz_m,
            qz_v=qz_v,
            z=z,
            untran_z=untran_z,
            ql_m=ql_msh,
            ql_v=ql_vsh,
            library_gene=library_gene,
            untran_l=untran_l,
            log_pro_back_mean=log_pro_back_mean,
        )

        ## Unshared RNA
        library_gene = latentrunsh["l"]
        untran_l = untran_latentrunsh["l"]
        
        px_ = self.rna_decoder_unshared(x_size,k,library_gene,batch_index,label)
        unshared_rna_dict = dict(
            px_scale=px_["scale"],
            px_r=self.px_r_us,
            px_rate=px_["rate"],
            px_dropout=px_["dropout"],
            qz_m=qz_m,
            qz_v=qz_v,
            z=z,
            ql_m=ql_mruns,
            ql_v=ql_vruns,
            library=library_gene,
        )

        ### Unshared protein
        py_, log_pro_back_mean = self.protein_decoder(k[gene_size:],None)
        py_["r"] = self.py_r_us
        unshared_protein_dict =  dict(
            py_= py_,
            qz_m = qz_m,
            qz_v = qz_v,
            z=z,
            untran_z=untran_z,
            log_pro_back_mean=log_pro_back_mean,
        )

        return shared_dict,unshared_rna_dict,unshared_protein_dict

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_unshared: torch.Tensor,
        y_unshared: torch.Tensor,
        local_l_mean_gene: torch.Tensor,
        local_l_var_gene: torch.Tensor,
        batch_index: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor
    ]:
        """ Returns the reconstruction loss and the Kullback divergences

        :param x: tensor of values with shape ``(batch_size, n_input_genes)``
        :param y: tensor of values with shape ``(batch_size, n_input_proteins)``
        :param local_l_mean_gene: tensor of means of the prior distribution of latent variable l
         with shape ``(batch_size, 1)````
        :param local_l_var_gene: tensor of variancess of the prior distribution of latent variable l
         with shape ``(batch_size, 1)``
        :param batch_index: array that indicates which batch the cells belong to with shape ``batch_size``
        :param label: tensor of cell-types labels with shape (batch_size, n_labels)
        :return: the reconstruction loss and the Kullback divergences
        """
        # Parameters for z latent distribution
        shared_dict,unshared_rna_dict,unshared_protein_dict = self.inference(x,y,x_unshared,y_unshared,batch_index,label)
    
        ## Globally lets talk bullshit
        qz_m = shared_dict["qz_m"]
        qz_v = shared_dict["qz_v"]
        z = shared_dict["z"]
        kl_div_z = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(0, 1)).sum(dim=1)
        z_gene = z[:x.shape[0]]
        z_protein = z[x.shape[0]:]

        ## Lets take shared features
        ql_m_shared = shared_dict["ql_m"]
        ql_v_shared = shared_dict["ql_v"]
        px_shared = shared_dict["px_"]
        py_shared = shared_dict["py_"]
        reconst_loss_gene_shared, reconst_loss_protein_shared = self.get_reconstruction_loss_shared(
            x, y, px_shared, py_shared, None
        )        
        kl_div_l_gene_shared = kl(
            Normal(ql_m_shared[:x.shape[0]], torch.sqrt(ql_v_shared[:x.shape[0]])),
            Normal(local_l_mean_gene, torch.sqrt(local_l_var_gene)),
        ).sum(dim=1)

        kl_div_back_pro_shared = kl(Normal(py_shared["back_alpha"], py_shared["back_beta"]), self.back_mean_prior_s).sum(dim=1)
        ## Lets take unshared features of RNA
        ql_m_unshared = unshared_rna_dict["ql_m"]
        ql_v_unshared = unshared_rna_dict["ql_v"]
        px_rate = unshared_rna_dict["px_rate"]
        px_r = unshared_rna_dict["px_r"]
        px_dropout = unshared_rna_dict["px_dropout"]
        reconst_loss_gene_unshared = self.get_reconstruction_loss_rna(x_unshared, px_rate, px_r, px_dropout)
        kl_div_l_gene_unshared = kl(
            Normal(ql_m_unshared[:x.shape[0]], torch.sqrt(ql_v_unshared[:x.shape[0]])),
            Normal(local_l_mean_gene, torch.sqrt(local_l_var_gene)),
        ).sum(dim=1)
        ## Lets take unshared features of Protein
        py_ = unshared_protein_dict["py_"]
        reconst_loss_protein_unshared = self.get_reconstruction_loss_protein(
            y_unshared, py_, None
        )
        kl_div_back_pro_unshared = kl(
            Normal(py_["back_alpha"], py_["back_beta"]), self.back_mean_prior_us
        ).sum(dim=1)

                
        return (
            reconst_loss_gene_unshared,
            reconst_loss_gene_shared,
            kl_div_l_gene_unshared,
            kl_div_l_gene_shared,
            reconst_loss_protein_unshared,
            reconst_loss_protein_shared,
            kl_div_back_pro_unshared,
            kl_div_back_pro_shared,
            kl_div_z,
            z_gene,
            z_protein,
        )



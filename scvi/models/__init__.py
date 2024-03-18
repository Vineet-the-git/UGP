from .classifier import Classifier
from .scanvi import SCANVI
from .vae import VAE, LDVAE
from .autozivae import AutoZIVAE
from .vaec import VAEC
from .jvae import JVAE
from .totalvi import TOTALVI
from .proteinvi_unshared import PROTENVI_UNSHARED

__all__ = [
    "SCANVI",
    "VAEC",
    "VAE",
    "LDVAE",
    "JVAE",
    "Classifier",
    "AutoZIVAE",
    "TOTALVI",
    "PROTENVI_UNSHARED",
]

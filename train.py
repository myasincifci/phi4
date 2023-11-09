from typing import Any
import numpy as np
from pytorch_lightning.utilities.types import STEP_OUTPUT
from pytorch_lightning.callbacks import ModelCheckpoint
import time

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

import nflows
from nflows.nn import nets as nets
from nflows.transforms.normalization import BatchNorm
from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.coupling import AdditiveCouplingTransform, AffineCouplingTransform, splines
from nflows.transforms.base import CompositeTransform
import matplotlib.pyplot as plt

from neulat.action import Phi4Action

from loss import ReparamKL
from neulat.models import Z2Nice, COUPLINGS

import hydra
from omegaconf import DictConfig, OmegaConf

from tqdm import tqdm

import matplotlib.pyplot as plt 
import pytorch_lightning as pl

import mlflow
from torch.optim.lr_scheduler import ReduceLROnPlateau

from pytorch_lightning.loggers import WandbLogger
import wandb

class Net(nn.Module):
    def __init__(self, num_id, num_trf) -> None:
        super().__init__()
        mid_dim=1000
        nblocks=4

        self.net = nn.Sequential(
            nn.Linear(num_id, mid_dim, bias=False),
            nn.Tanh(),
            nn.Sequential(*[
                nn.Sequential(
                    nn.Linear(mid_dim, mid_dim, bias=False),
                    nn.Tanh(),
                ) for _ in range(nblocks)
            ]),
            nn.Linear(mid_dim, num_trf, bias=False)
        )
    
    def forward(self, x, ctx):
        return self.net(x)
    
class Z2NICE(AdditiveCouplingTransform):
    def __init__(self, mask, transform_net_create_fn, unconditional_transform=None, scale_activation=...):
        super().__init__(mask, transform_net_create_fn, unconditional_transform, scale_activation)

        self.scale = nn.Parameter(
            torch.zeros((1, torch.tensor(list(mask.shape)).prod()//2)),
            requires_grad=True
        )

    def _scale_and_shift(self, transform_params):
        shift = transform_params
        scale = torch.exp(self.scale)
        return scale, shift
    
class SimpleRealNVP(Flow):
    """An simplified version of Real NVP for 1-dim inputs.

    This implementation uses 1-dim checkerboard masking but doesn't use multi-scaling.

    Reference:
    > L. Dinh et al., Density estimation using Real NVP, ICLR 2017.
    """

    def __init__(
        self,
        features,
        hidden_features,
        num_layers,
        num_blocks_per_layer,
        use_volume_preserving=False,
        activation=F.relu,
        dropout_probability=0.0,
        batch_norm_within_layers=False,
        batch_norm_between_layers=False,
    ):

        if use_volume_preserving:
            coupling_constructor = AdditiveCouplingTransform
            coupling_constructor = Z2NICE
            # coupling_constructor = splines.linear_spline
        else:
            coupling_constructor = AffineCouplingTransform

        mask = torch.ones(features)
        mask[::2] = -1

        def create_resnet(in_features, out_features):
            return nets.ResidualNet(
                in_features,
                out_features,
                hidden_features=hidden_features,
                num_blocks=num_blocks_per_layer,
                activation=activation,
                dropout_probability=dropout_probability,
                use_batch_norm=batch_norm_within_layers,
            )

        layers = []
        for _ in range(num_layers):
            transform = coupling_constructor(
                mask=mask, transform_net_create_fn=create_resnet
            )
            layers.append(transform)
            mask *= -1
            if batch_norm_between_layers:
                layers.append(BatchNorm(features=features))

        super().__init__(
            transform=CompositeTransform(layers),
            distribution=StandardNormal([features]),
        )

    
class NicePhi4(pl.LightningModule):
    def __init__(self, cfg, lat_shape, dev, kappa=0.3, lambd=0.022,*args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.cfg = cfg
        
        # self.lattice_shape = lat_shape
        self.action=Phi4Action(kappa, lambd)

        # Define flow
        num_layers = 6
        base_dist = StandardNormal(shape=[128]).to(dev)

        transforms = []
        for i in range(num_layers):
            transforms.append(
                Z2NICE(
                    self._make_checker_mask(self.lattice_shape, i % 2), 
                    Net
                ),
            )
        transform = CompositeTransform(transforms)

        self.flow = Flow(transform, base_dist)

        self.criterion = ReparamKL(
            model=self.flow, 
            action=self.action, 
            lat_shape=self.lattice_shape,
            batch_size=20_000
        )

    def training_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        loss, loss_summands, action = self.criterion()
        loss_var = loss_summands.var()

        # mlflow.log_metric("loss", loss, )
        # mlflow.log_metric("loss_var", loss_var)

        self.log_dict({
            "loss": loss,
            "var": loss_var
        }, prog_bar=True)

        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.flow.parameters(), lr=self.cfg.param.lr, amsgrad=True)
        lr_scheduler = ReduceLROnPlateau(
            optimizer,
            factor=0.92,
            verbose=True,
            patience=3000,
            threshold=1e-4,
            min_lr=1e-7
        )

        return [optimizer]
    
    def _make_checker_mask(self, shape, parity):
        checker = torch.ones(shape, dtype=torch.uint8) - parity
        checker[::2, ::2] = parity
        checker[1::2, 1::2] = parity

        return checker.flatten()
    
class RealNVP(pl.LightningModule):
    def __init__(self, cfg, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        
        self.lattice_shape = cfg.lat_shape
        self.action=Phi4Action(cfg.kappa, cfg.lambd)

        # Define flow
        self.flow = SimpleRealNVP(
            features=16*8,
            hidden_features=100,
            num_layers=20,
            num_blocks_per_layer=4, 
            use_volume_preserving=True,
        )
        
        self.criterion = ReparamKL(
            model=self.flow, 
            action=self.action, 
            lat_shape=self.lattice_shape,
            batch_size=cfg.param.batch_size,
        )

    def training_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        loss, loss_summands, action = self.criterion()
        loss_var = loss_summands.var()

        # mlflow.log_metric("loss", loss, )
        # mlflow.log_metric("loss_var", loss_var)

        self.log_dict({
            "loss": loss,
            "var": loss_var
        }, prog_bar=True)

        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.flow.parameters(), lr=5e-4, amsgrad=True)
        lr_scheduler = ReduceLROnPlateau(
            optimizer,
            factor=0.92,
            verbose=True,
            patience=3000,
            threshold=1e-4,
            min_lr=1e-7
        )

        return [optimizer]
    
    def _make_checker_mask(self, shape, parity):
        checker = torch.ones(shape, dtype=torch.uint8) - parity
        checker[::2, ::2] = parity
        checker[1::2, 1::2] = parity

        return checker.flatten()

class DummyDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()

    def __len__(self):
        return 100
    
    def __getitem__(self, index) -> Any:
        return torch.tensor([0])

@hydra.main(version_base=None, config_path="configs")
def main(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))    # start a new wandb run to track this script
    
    logger = True
    if cfg.logging:
        # start a new wandb run to track this script
        wandb.login(
            key="deeed2a730495791be1a0158cf49240b65df1ffa"
        )
        wandb.init(
            # set the wandb project where this run will be logged
            project="Phi-4",
            
            # track hyperparameters and run metadata
            config={
                "learning_rate": cfg.param.lr,

                "batch_size": cfg.param.batch_size,

                "architecture": "ResNet 50",
                "dataset": "camelyon17",
            },

            id=f"{cfg.name}-{time.time()}"
        )
        logger = WandbLogger()

    ############################################################################

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    train_set = DummyDataset()
    train_loader = DataLoader(train_set)

    checkpoint_callback = ModelCheckpoint(dirpath=".", save_top_k=5, monitor="loss")

    trainer = pl.Trainer(
        max_steps=1_000_000,
        accelerator="auto",
        callbacks=[checkpoint_callback]
    )

    # model = NicePhi4(lat_shape=(16,8), dev=device)
    model = RealNVP(cfg=cfg)

    trainer.fit(
        model=model,
        train_dataloaders=train_loader
    )

if __name__ == "__main__":
    main()
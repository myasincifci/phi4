from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT
from pytorch_lightning.callbacks import ModelCheckpoint
import time

import torch
from torch.utils.data import Dataset, DataLoader

from neulat.action import Phi4Action
from loss import ReparamKL

import hydra
from omegaconf import DictConfig, OmegaConf

import pytorch_lightning as pl

from torch.optim.lr_scheduler import ReduceLROnPlateau

from pytorch_lightning.loggers import WandbLogger
import wandb

from models import model_dict

    
class NeulatModule(pl.LightningModule):
    def __init__(
            self, 
            cfg, 
            criterion,
            *args: Any, 
            **kwargs: Any
    ) -> None:
    
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        
        self.criterion = criterion

    def training_step(self, _) -> STEP_OUTPUT:
        loss, loss_summands, _ = self.criterion()
        loss_var = loss_summands.var()

        self.log_dict({
            "loss": loss,
            "var": loss_var
        }, prog_bar=True)

        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-4, amsgrad=True)
        lr_scheduler = ReduceLROnPlateau(
            optimizer,
            factor=0.92,
            verbose=True,
            patience=3000,
            threshold=1e-4,
            min_lr=1e-7
        )

        return [optimizer]

class DummyDataset(Dataset):
    """Dummy dataset to pass to Pytorch Ligtning.
    Returns zero tensor"""

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

    # flow = SimpleRealNVP(
    #     features=16*8,
    #     hidden_features=100,
    #     num_layers=20,
    #     num_blocks_per_layer=4, 
    #     use_volume_preserving=True,
    # )

    # flow2 = Z2Nice(
    #     lat_shape=list(cfg.lat_shape),
    #     hidden_features=1000,
    #     num_layers=4
    # )

    flow = model_dict[cfg.model.type](
        list(cfg.lat_shape),
        **dict(cfg.model.kwargs)
    )

    action = Phi4Action(cfg.kappa, cfg.lambd)
    criterion = ReparamKL(
        model=flow, 
        action=action, 
        lat_shape=cfg.lat_shape,
        batch_size=cfg.param.batch_size,
    )    

    model = NeulatModule(
        cfg=cfg,
        criterion=criterion   
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_loader
    )

if __name__ == "__main__":
    main()
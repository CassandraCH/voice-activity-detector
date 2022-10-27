import torch
import pytorch_lightning as pl
import torch.nn.functional as F

from torch.utils.data import DataLoader, random_split
from dataloader_example import m2set
from tcn import TCN



class TcnModel(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        x_hat = self.model(x)
        #loss = F.binary_cross_entropy(x_hat[:, 0, :], y)
        loss = F.cross_entropy(x_hat, y)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


m2s = m2set(cutset_file="lists/allies_train_fbank_vad.jsonl.gz")
train_loader = DataLoader(m2s)





# train model
trainer = pl.Trainer()
tcnmodel = TcnModel(TCN(80, [50, 50, 50, 2]))
trainer.fit(model=tcnmodel, train_dataloaders=train_loader)

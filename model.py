from pytorch_lightning import LightningDataModule, LightningModule, Trainer
import torch.nn as nn
import dgl
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import torch

class SAGE(LightningModule):
    """Multi-layer GraphSAGE lightning module for node classification task."""

    def __init__(self, in_feats: int, n_layers: int, n_hidden: int, n_classes: int, aggregator_type: str):
        super().__init__()
        self.save_hyperparameters()

        self.layers = nn.ModuleList()
        self.layers.append(dgl.nn.SAGEConv(in_feats, n_hidden, aggregator_type))
        for i in range(1, n_layers - 1):
            self.layers.append(dgl.nn.SAGEConv(n_hidden, n_hidden, aggregator_type))

        self.layers.append(dgl.nn.SAGEConv(n_hidden, n_classes, aggregator_type))

        self.dropout = nn.Dropout(0.5)
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.train_acc = 0
        self.val_acc = 0

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h
        
    def training_step(self, batch, batch_idx):
        input_nodes, output_nodes, blocks = batch
        x = blocks[0].srcdata["feat"]
        y = blocks[-1].dstdata["label"]
        y_hat = self(blocks, x)
        loss = F.cross_entropy(y_hat, y)
        self.train_acc=accuracy_score(torch.argmax(y_hat, 1).cpu(), y.cpu())
        self.log(
            "train_acc",
            self.train_acc,
            prog_bar=True,
            on_step=True,
            on_epoch=False
        )
        return loss

    def validation_step(self, batch, batch_idx):
        input_nodes, output_nodes, blocks = batch
        x = blocks[0].srcdata["feat"]
        y = blocks[-1].dstdata["label"]
        y_hat = self(blocks, x)
        self.val_acc=accuracy_score(torch.argmax(y_hat, 1).cpu(), y.cpu())
        self.log(
            "val_acc",
            self.val_acc,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=0.001,
            weight_decay=5e-4
        )
        return optimizer
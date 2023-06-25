import dgl
from pytorch_lightning import LightningDataModule

class DataModule(LightningDataModule):

    def __init__(
        self, graph, train_idx, val_idx, fanouts, batch_size, n_classes, device
    ):
        super().__init__()

        sampler = dgl.dataloading.NeighborSampler(
            fanouts, prefetch_node_feats=["feat"], prefetch_labels=["label"]
        )

        self.graph = graph
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.sampler = sampler
        self.batch_size = batch_size
        self.in_feats = graph.ndata["feat"].shape[1]
        self.n_classes = n_classes
        self.device=device

    def train_dataloader(self):
        return dgl.dataloading.DataLoader(
            self.graph,
            self.train_idx.to(self.device),
            self.sampler,
            device=self.device,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0,
            use_uva=True
        )

    def val_dataloader(self):
        return dgl.dataloading.DataLoader(
            self.graph,
            self.val_idx.to(self.device),
            self.sampler,
            device=self.device,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0,
            use_uva=True,
        )
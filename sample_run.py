from ogb.nodeproppred import DglNodePropPredDataset
import pytorch_lightning as pl
import os
from optuna.integration import PyTorchLightningPruningCallback
from utils import DictLogger
from dataloader import DataModule
import torch
from model import SAGE

import warnings
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import logging
logging.getLogger('lightning').setLevel(0)
import optuna

dataset = DglNodePropPredDataset("ogbn-products")
graph, labels = dataset[0]

graph.ndata["label"] = labels.squeeze()

split_idx = dataset.get_idx_split()

train_idx, val_idx, test_idx = (
    split_idx["train"],
    split_idx["valid"],
    split_idx["test"],
)

def objective(trial):
    MODEL_DIR="./logs"
    # PyTorch Lightning will try to restore model parameters from previous trials if checkpoint
    # filenames match. Therefore, the filenames for each trial must be made unique.
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        os.path.join(MODEL_DIR, "trial_{}".format(trial.number)), monitor="val_acc",save_top_k=1
    )

    # The default logger in PyTorch Lightning writes to event files to be consumed by
    # TensorBoard. We create a simple logger instead that holds the log in memory so that the
    # final accuracy can be obtained after optimization. When using the default logger, the
    # final accuracy could be stored in an attribute of the `Trainer` instead.
    logger = DictLogger(save_dir="logs")
    trainer = pl.Trainer(
    logger=logger,
    accelerator='gpu',
    devices=[1],
    max_epochs=10,
    # note, we purpose-fully disabled the progress bar to prevent flooding our notebook's console
    # in normal settings, we can/should definitely turn it on
    enable_progress_bar=False,
    log_every_n_steps=100,
    # callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_acc")],
    callbacks=[checkpoint_callback]
    )
    # params = {
    #           'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-1),
    #           'optimizer': trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"]),
    #           'n_unit': trial.suggest_int("n_unit", 4, 18)
    #           }
    n_hidden = trial.suggest_int("n_hidden", 16, 256)
    fanouts = [trial.suggest_int("fanouts", 5, 15), trial.suggest_int("n_unit", 5, 15), trial.suggest_int("n_unit", 5, 15)]
    aggregator_type = trial.suggest_categorical("aggregator_type", ["mean", "pool", "lstm"])
    batch_size = trial.suggest_int("batch_size", 16, 1024)
    n_layers = len(fanouts)

    data_module = DataModule(graph, train_idx, val_idx, fanouts, batch_size, dataset.num_classes, device)
    model = SAGE(
    in_feats=data_module.in_feats,
    n_layers=n_layers,
    n_hidden=n_hidden,
    n_classes=data_module.n_classes,
    aggregator_type=aggregator_type
    )

    trainer.fit(model,datamodule=data_module)

    print(logger.metrics[-1].keys())

    return logger.metrics[-1]["val_acc_epoch"]


study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=2)

from pytorch_lightning.loggers import TensorBoardLogger

class DictLogger(TensorBoardLogger):
    """PyTorch Lightning `dict` logger."""

    # see https://github.com/PyTorchLightning/pytorch-lightning/blob/50881c0b31/pytorch_lightning/logging/base.py

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics = []

    def log_metrics(self, metrics, step=None):
        """Logs the training metrics

        :param metrics: the values of the metrics
        :type metrics: dict
        :param step: the ID of the current epoch, defaults to None
        :type step: int, optional
        """
        super().log_metrics(metrics, step=step)
        self.metrics.append(metrics)


from pytorch_lightning import LightningModule, LightningDataModule, Trainer
from pytorch_lightning.callbacks import TQDMProgressBar
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


class TorchWrapper(LightningModule):
    def __init__(self, model, loss, metric):
        super().__init__()

        self.model = model

        self.metric = metric
        self.loss = loss

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def _common_step(self, batch):
        X, y = batch
        logits = self(X)

        loss = self.loss(logits, y)
        metric = self.metric(logits, y)

        return loss, metric

    def training_step(self, batch, batch_idx):

        loss, metric = self._common_step(batch)

        self.log("train_loss", loss, prog_bar=False, on_epoch=True, on_step=False)
        self.log("train_metric", metric, prog_bar=True, on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, metric = self._common_step(batch)

        self.log("val_loss", loss, prog_bar=False, on_epoch=True, on_step=False)
        self.log("val_metric", metric, prog_bar=True, on_epoch=True, on_step=False)

    def test_step(self, batch, batch_idx):
        loss, metric = self._common_step(batch)

        self.log("test_loss", loss, prog_bar=False, on_epoch=True, on_step=False)
        self.log("test_metric", metric, prog_bar=False, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        return Adam(self.parameters())


class MNISTDataModule(LightningDataModule):
    def __init__(self, batch_size=64):
        super().__init__()
        self.batch_size = batch_size
        self.tansforms = transforms.Compose([transforms.ToTensor()])

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        # download only, run once
        datasets.MNIST(".", train=True, download=True, transform=self.tansforms)
        datasets.MNIST(".", train=False, download=True, transform=self.tansforms)

    def setup(self, stage=None):
        mnist_train = datasets.MNIST(".", train=True, download=False, transform=self.tansforms)
        mnist_test = datasets.MNIST(".", train=False, download=False, transform=self.tansforms)

        mnist_train, mnist_val = random_split(mnist_train, [55000, 5000])

        self.train_dataset = mnist_train
        self.val_dataset = mnist_val
        self.test_dataset = mnist_test

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=2)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=2)


def get_default_trainer(**kwargs):
    kwargs.setdefault('max_epochs', 5)
    kwargs.setdefault('gpus', 1)
    kwargs.setdefault('logger', False)
    kwargs.setdefault('num_sanity_val_steps', 0)
    kwargs.setdefault('enable_checkpointing', False)
    kwargs.setdefault('enable_model_summary', False)
    kwargs.setdefault('callbacks', [TQDMProgressBar(refresh_rate=20)])

    return Trainer(**kwargs)

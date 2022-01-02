import torch
from torchmetrics import MeanMetric
from torch.utils.data import TensorDataset, DataLoader, random_split
from tqdm.auto import tqdm

from ML_tools.aliases import get_loss, get_metric, get_optimizer, camel_to_snake


class Trainer:
    def __init__(self, model, loss, metric, optimizer, device='auto'):

        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        if isinstance(loss, str):
            self.loss = get_loss(loss)
        else:
            self.loss = loss

        if isinstance(optimizer, str):
            self.optimizer = get_optimizer(optimizer, model.parameters())
        else:
            self.optimizer = optimizer

        if isinstance(metric, str):
            self.train_metric = get_metric(metric).to(self.device)
        else:
            self.train_metric = metric
        self.val_metric = self.train_metric.clone()

        self.model = model.to(self.device)

        self.train_loss_metric = MeanMetric().to(self.device)
        self.val_loss_metric = MeanMetric().to(self.device)

        self.metric_name = camel_to_snake(type(self.train_metric).__name__)
        self.validate = True
        self.history = {'train_loss': [], f'train_{self.metric_name}': [],
                        'val_loss': [], f'val_{self.metric_name}': []}

    def fit(self,
            x_train,
            y_train=None,
            epochs=1,
            batch_size=1,
            validation_data=None,
            validation_split=0,
            verbose=2):

        if isinstance(x_train, DataLoader):
            return self._fit_dataloader(x_train, validation_data, epochs, verbose)

        train_set = TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train))
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = None

        if validation_data is not None:
            val_set = TensorDataset(torch.Tensor(validation_data[0]),
                                    torch.Tensor(validation_data[1]))
            val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
        else:
            if 0 <= validation_split <= 1:
                # create validation dataset
                validation_split = min(validation_split, 1 - validation_split)
                n_val = int(len(train_set) * validation_split)
                n_train = len(train_set) - n_val

                train_set, val_set = random_split(train_set, (n_train, n_val))
                val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

        return self._fit_dataloader(train_loader, val_loader, epochs, verbose)

    def _fit_dataloader(self, train_loader, val_loader=None, epochs=1,verbose=2):
        if val_loader is None:
            self.validate = False

        for epoch in tqdm(range(epochs), disable=(verbose != 1), desc="Epochs:"):
            self.model.train()
            pbar = tqdm(train_loader, disable=(verbose <= 1))
            for x_batch, y_batch in pbar:
                pbar.set_description(f"Epoch {epoch + 1}/{epochs}")
                self._trainig_step(x_batch.to(self.device), y_batch.to(self.device))

            if self.validate:
                self.model.eval()
                for x_batch, y_batch in val_loader:
                    self._validation_step(x_batch.to(self.device), y_batch.to(self.device))

            self._log_progress(verbose)

        self.model.eval()
        return self.history

    def _base_step(self, x_batch, y_batch):
        y_pred = self.model(x_batch)
        loss_value = self.loss(y_pred, y_batch)

        if y_pred.ndim > 1:
            y_pred = y_pred.argmax(dim=-1)

        return y_pred, loss_value

    def _trainig_step(self, x_batch, y_batch):
        y_pred, loss_value = self._base_step(x_batch, y_batch)

        self.train_loss_metric.update(loss_value)
        self.train_metric.update(y_batch, y_pred)

        loss_value.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def _validation_step(self, x_batch, y_batch):
        y_pred, loss_value = self._base_step(x_batch, y_batch)

        self.val_loss_metric.update(loss_value)
        self.val_metric.update(y_batch, y_pred)

    def evaluate(self, x_test, y_test=None, verbose=1):
        if isinstance(x_test, DataLoader):
            return self._evaluate_dataloader(x_test, verbose)

        test_set = TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test))
        test_loader = DataLoader(test_set, batch_size=32, shuffle=True)

        return self._evaluate_dataloader(test_loader, verbose)

    def _evaluate_dataloader(self, test_loader, verbose=1):

        self.val_loss_metric.reset()
        self.val_metric.reset()

        self.model.eval()
        for x_batch, y_batch in tqdm(test_loader, disable=(verbose <= 0)):
            self._validation_step(x_batch.to(self.device), y_batch.to(self.device))

        loss = float(self.val_loss_metric.compute())
        self.val_loss_metric.reset()

        metric = float(self.val_metric.compute())
        self.val_metric.reset()

        if verbose >= 1:
            message = f"loss: {loss:.3f} - {self.metric_name}: {metric:.3f}"
            print(message)

        return (loss, metric)

    def _log_progress(self, verbose):
        train_loss = float(self.train_loss_metric.compute())
        self.history.get('train_loss').append(train_loss)
        self.train_loss_metric.reset()

        train_metric = float(self.train_metric.compute())
        self.history.get(f'train_{self.metric_name}').append(train_metric)
        self.train_metric.reset()

        if self.validate:
            val_loss = float(self.val_loss_metric.compute())
            self.history.get('val_loss').append(val_loss)
            self.val_loss_metric.reset()

            val_metric = float(self.val_metric.compute())
            self.history.get(f'val_{self.metric_name}').append(val_metric)
            self.val_metric.reset()

        if verbose >= 2:
            message = f"loss: {train_loss:.3f} - {self.metric_name}: {train_metric:.3f}"
            if self.validate:
                message += f" - val_loss: {val_loss:.3f} - val_{self.metric_name}: {val_metric:.3f}"
            print(message)

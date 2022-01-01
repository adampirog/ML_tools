from numpy import ndarray, float32
from torch.utils.data import TensorDataset, DataLoader, random_split
from torchmetrics import MeanMetric
from tqdm.auto import tqdm
import torch

from aliases import get_loss, get_metric, get_optimizer, camel_to_snake

class Trainer():
    def __init__(self, model, loss, metric, optimizer, device='auto'):

        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.model = model.to(self.device)

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

        self.train_loss_metric = MeanMetric().to(self.device)
        self.val_loss_metric = MeanMetric().to(self.device)

        self.metric_name = camel_to_snake(type(self.train_metric).__name__)
        self.validate = True
        self.history = {'train_loss': [], f'train_{self.metric_name}': [],
                        'val_loss': [], f'val_{self.metric_name}': []}


    def fit(self, x_train, y_train,
            epochs=1, *, batch_size=1,
            validation_data=None, validation_split=0, verbose=2):

        if not isinstance(x_train, ndarray):
            if isinstance(x_train, DataLoader):
                raise Exception("Type missmatch, consider using fit_dataloader function")
            else:
                raise Exception("Type missmatch")

        train_set = TensorDataset(torch.from_numpy(x_train.astype(float32)),
                                  torch.from_numpy(y_train.astype(float32).squeeze()))

        if validation_data is not None:
            # validation provided
            assert len(validation_data) == 2
            val_set = TensorDataset(torch.from_numpy(validation_data[0].astype(float32)),
                                    torch.from_numpy(validation_data[1].astype(float32)))
            val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
        else:
            # validation not provided
            if validation_split == 0:
                # perform no validation
                val_loader = None
            else:
                # split the data
                assert 0 <= validation_split <= 1
                validation_split = min(validation_split, 1 - validation_split)
                n_val = int(len(train_set) * validation_split)
                n_train = int(len(train_set) - n_val)

                train_set, val_set = random_split(train_set, (n_train, n_val))
                val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        return self.fit_dataloader(train_loader, val_loader, epochs, verbose)


    def fit_dataloader(self, train_loader, val_loader, epochs, verbose=2):
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

            self.log_progress(verbose)

        self.model.eval()
        return self.history


    def _base_step(self, x_batch, y_batch):
        y_pred = self.model(x_batch).squeeze()
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

    def evaluate(self, x_test, y_test, verbose=1):
        if not isinstance(x_test, ndarray):
            if isinstance(x_test, DataLoader):
                raise Exception("Type missmatch, consider using evaluate_dataloader function")
            else:
                raise Exception("Type missmatch")

        test_set = TensorDataset(torch.from_numpy(x_test.astype(float32)),
                                  torch.from_numpy(y_test.astype(float32).squeeze()))
        test_loader = DataLoader(test_set, batch_size=32, shuffle=True)

        return self.evaluate_dataloader(test_loader, verbose)


    def evaluate_dataloader(self, test_loader, verbose=1):

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
            progress_message = f"loss: {loss:.3f} - {self.metric_name}: {metric:.3f}"
            print(progress_message)

        return (loss, metric)



    def log_progress(self, verbose):
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
            progress_message = f"loss: {train_loss:.3f} - {self.metric_name}: {train_metric:.3f}"
            if self.validate:
                progress_message += f" - val_loss: {val_loss:.3f} - val_{self.metric_name}: {val_metric:.3f}"
            print(progress_message)

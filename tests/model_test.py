import numpy as np
from torch import nn
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from torch_tools import Trainer


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(100, 50)
        self.layer_2 = nn.Linear(50, 10)
        self.layer_3 = nn.Linear(10, 1)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.layer_1(x)
        x = self.act(x)
        x = self.layer_2(x)
        x = self.act(x)
        x = self.layer_3(x)
        return x.view(len(x),)


def run():
    X, y = make_regression(n_samples=20_000, n_features=100, n_informative=100,
                           n_targets=1, noise=0.2, random_state=101)
    X, y = X.astype(np.float32), y.astype(np.float32)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = MyModel()
    trainer = Trainer(model, loss="mse", optimizer="adam", metric="mae")

    trainer.fit(X_train, y_train, epochs=2, batch_size=32, validation_split=0.2)

    trainer.evaluate(X_test, y_test)

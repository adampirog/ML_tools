import re
import torchmetrics
from torch import nn, optim

OPTIMIZERS = {
    "adam": "Adam",
    "Adam": "Adam",

    "rmsprop": "RMSprop",
    "rms": "RMSprop",
    "RMSprop": "RMSprop"
}

LOSSES = {
    "crossentropy": "CrossEntropyLoss",
    "CrossEntropyLoss": "CrossEntropyLoss",
    "CrossEntropy": "CrossEntropyLoss",

    "MSELoss": "MSELoss",
    "MSE": "MSELoss",
    "mse": "MSELoss",
}

METRICS = {
    "acc": "Accuracy",
    "accuracy": "Accuracy",
    "Accuracy": "Accuracy",

    "MeanAbsoluteError": "MeanAbsoluteError",
    "meanabsoluteerror": "MeanAbsoluteError",
    "mae": "MeanAbsoluteError",

    "MeanSquaredError": "MeanSquaredError",
    "meansquarederror": "MeanSquaredError",
    "mse": "MeanSquaredError"
}

def get_optimizer(alias, model_parameters):
    class_name = OPTIMIZERS.get(alias)

    if not class_name:
        raise ValueError(f"Couldn't resolve optimizer alias: {alias}")

    clss = getattr(optim, class_name)
    return clss(model_parameters)

def get_metric(alias):
    class_name = METRICS.get(alias)

    if not class_name:
        raise ValueError(f"Couldn't resolve metric alias: {alias}")

    clss = getattr(torchmetrics, class_name)
    return clss()

def get_loss(alias):
    class_name = LOSSES.get(alias)

    if not class_name:
        raise ValueError(f"Couldn't resolve loss alias: {alias}")

    clss = getattr(nn, class_name)
    return clss()

def camel_to_snake(name):
    return re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()

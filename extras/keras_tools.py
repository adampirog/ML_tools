import datetime
import logging
import os

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras


SEED = 2021
DEVICE = "GPU"

np.random.seed(SEED)
tf.random.set_seed(SEED)
keras.backend.clear_session()

logging.basicConfig(format='%(levelname)s: %(message)s')

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


if DEVICE == 'GPU':
    gpu = tf.config.list_physical_devices('GPU')
    assert gpu
    tf.config.experimental.set_memory_growth(gpu[0], True)

elif DEVICE == 'TPU':
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    assert tf.config.list_logical_devices('TPU')

else:
    tf.config.set_visible_devices([], 'GPU')
    tf.config.set_visible_devices([], 'TPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type not in ['GPU', 'TPU']


# %load_ext tensorboard
# %tensorboard --logdir=./logs --port=6006
def get_logdir(root_logdir=None):
    if not root_logdir:
        root_logdir = os.path.join(os.curdir, "logs")
    run_id = datetime.datetime.now().strftime("run_%Y-%m-%d_%H:%M:%S")
    return os.path.join(root_logdir, run_id)


def plot_training(history, limit_grid=None, filepath=""):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    if limit_grid:
        plt.gca().set_ylim(*limit_grid)
    if filepath:
        plt.savefig(filepath)
    plt.show()


def mount_drive():
    from google.colab import drive
    drive.mount('/content/drive')

    return '/content/drive/My Drive/'

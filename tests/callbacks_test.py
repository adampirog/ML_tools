import unittest
from unittest.mock import Mock

from torch_tools.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint


class TestEarlyStopping(unittest.TestCase):

    def test_monitor(self):
        callback = EarlyStopping(patience=10, monitor="val_loss")

        trainer = Mock()
        trainer.history = {"training_loss": [1], "training_acc": [1]}

        with self.assertRaises(ValueError):
            callback.on_epoch_end(trainer)

    def test_init(self):
        callback = EarlyStopping(patience=10, monitor="val_loss", restore_best_weights=True)

        trainer = Mock()
        trainer.history = {"val_loss": [2]}
        trainer.model.state_dict = Mock(return_value=[1, 2, 3])

        callback.on_epoch_end(trainer)

        self.assertEqual(callback.best_value, 2)
        self.assertEqual(callback.best_model, [1, 2, 3])

    def test_min_mode(self):
        callback = EarlyStopping(patience=10, monitor="val_loss", mode="min", restore_best_weights=False)
        trainer = Mock()

        trainer.history = {"val_loss": [2]}
        callback.on_epoch_end(trainer)
        trainer.history = {"val_loss": [1]}
        callback.on_epoch_end(trainer)
        trainer.history = {"val_loss": [5]}
        callback.on_epoch_end(trainer)
        trainer.history = {"val_loss": [2]}
        callback.on_epoch_end(trainer)

        self.assertEqual(callback.best_value, 1)

    def test_max_mode(self):
        callback = EarlyStopping(patience=10, monitor="val_loss", mode="max", restore_best_weights=False)
        trainer = Mock()

        trainer.history = {"val_loss": [5]}
        callback.on_epoch_end(trainer)
        trainer.history = {"val_loss": [1]}
        callback.on_epoch_end(trainer)
        trainer.history = {"val_loss": [2]}
        callback.on_epoch_end(trainer)
        trainer.history = {"val_loss": [3]}
        callback.on_epoch_end(trainer)

        self.assertEqual(callback.best_value, 5)

    def test_not_restoring_weights(self):
        callback = EarlyStopping(patience=2, restore_best_weights=False)
        trainer = Mock()

        trainer.history = {"val_loss": [2]}
        self.assertEqual(callback.on_epoch_end(trainer), False)
        self.assertEqual(callback.no_steps, 0)

        trainer.history = {"val_loss": [3]}
        self.assertEqual(callback.on_epoch_end(trainer), False)
        self.assertEqual(callback.no_steps, 1)

        trainer.history = {"val_loss": [1]}
        self.assertEqual(callback.on_epoch_end(trainer), False)
        self.assertEqual(callback.no_steps, 0)

        trainer.history = {"val_loss": [5]}
        self.assertEqual(callback.on_epoch_end(trainer), False)
        self.assertEqual(callback.no_steps, 1)

        trainer.history = {"val_loss": [6]}
        self.assertEqual(callback.on_epoch_end(trainer), True)
        self.assertEqual(callback.no_steps, 2)

        self.assertEqual(callback.best_model, None)
        self.assertEqual(trainer.model.load_state_dict.called, False)

    def test_restoring_weights(self):
        callback = EarlyStopping(patience=2, restore_best_weights=True)
        trainer = Mock()

        trainer.history = {"val_loss": [2]}
        trainer.model.state_dict = Mock(return_value=[1])
        self.assertEqual(callback.on_epoch_end(trainer), False)
        self.assertEqual(callback.no_steps, 0)

        trainer.history = {"val_loss": [3]}
        trainer.model.state_dict = Mock(return_value=[2])
        self.assertEqual(callback.on_epoch_end(trainer), False)
        self.assertEqual(callback.no_steps, 1)

        trainer.history = {"val_loss": [1]}
        trainer.model.state_dict = Mock(return_value=[3])
        self.assertEqual(callback.on_epoch_end(trainer), False)
        self.assertEqual(callback.no_steps, 0)

        trainer.history = {"val_loss": [5]}
        trainer.model.state_dict = Mock(return_value=[4])
        self.assertEqual(callback.on_epoch_end(trainer), False)
        self.assertEqual(callback.no_steps, 1)

        trainer.history = {"val_loss": [6]}
        trainer.model.state_dict = Mock(return_value=[5])
        self.assertEqual(callback.on_epoch_end(trainer), True)
        self.assertEqual(callback.no_steps, 2)

        self.assertEqual(callback.best_model, [3])
        self.assertEqual(trainer.model.load_state_dict.call_count, 1)


class TestModelCheckpoint(unittest.TestCase):

    def test_monitor(self):
        callback = EarlyStopping(patience=10, monitor="val_loss")

        trainer = Mock()
        trainer.history = {"training_loss": [1], "training_acc": [1]}

        with self.assertRaises(ValueError):
            callback.on_epoch_end(trainer)

    def test_filenames(self):
        callback = ModelCheckpoint(filepath="model_{epoch}.h5", save_best_only=False)

        trainer = Mock()
        trainer.history = {"train_loss": [1, 2]}

        string = callback._parse_filepath(trainer)
        self.assertEqual(string, "model_2.h5")

        callback = ModelCheckpoint(filepath="model_{epoch}_{train_loss}.h5", save_best_only=False)

        trainer = Mock()
        trainer.history = {"train_loss": [99, 78]}

        string = callback._parse_filepath(trainer)
        self.assertEqual(string, "model_2_78.h5")

        callback = ModelCheckpoint(filepath="model_{vall_loss}_{train_loss}.h5", save_best_only=False)

        trainer = Mock()
        trainer.history = {"train_loss": [99, 78],
                           "vall_loss": [55, 45]}

        string = callback._parse_filepath(trainer)
        self.assertEqual(string, "model_45_78.h5")


class CSVLoggerTest(unittest.TestCase):

    def test_file_creation(self):
        callback = CSVLogger("test.csv")
        trainer = Mock()
        trainer.history = {"train_loss": [],
                           "vall_loss": []}

        callback._create_file(trainer)

    def test_new_file(self):
        callback = CSVLogger("test.csv")
        trainer = Mock()

        trainer.history = {"train_loss": [99],
                           "vall_loss": [55]}

        callback.on_epoch_end(trainer)

    def test_iters(self):
        callback = CSVLogger("test.csv")
        trainer = Mock()

        trainer.history = {"train_loss": [99],
                           "vall_loss": [55]}

        callback.on_epoch_end(trainer)

        trainer.history = {"train_loss": [99, 33],
                           "vall_loss": [55, 12]}

        callback.on_epoch_end(trainer)

    def test_append(self):
        callback = CSVLogger("test.csv", append=True)
        trainer = Mock()

        trainer.history = {"train_loss": [11],
                           "vall_loss": [12]}

        callback.on_epoch_end(trainer)


if __name__ == '__main__':
    unittest.main()

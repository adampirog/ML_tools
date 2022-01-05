from torch import save as torch_save


class EarlyStopping:
    def __init__(self,
                 patience,
                 monitor="val_loss",
                 min_delta=0,
                 mode="min",
                 restore_best_weights=True):

        self.monitor = monitor
        self.min_delta = min_delta
        self.mode = mode
        self.patience = patience
        self.restore_best_weights = restore_best_weights

        self.best_value = None
        self.best_model = None
        self.no_steps = 0

    def on_epoch_end(self, trainer):
        # first time called
        if not self.best_value:
            self._init_params(trainer)
            return self._stop(trainer)

        new_value = trainer.history.get(self.monitor)[-1]

        if self.mode == "max":
            delta = new_value - self.best_value
        else:
            delta = self.best_value - new_value

        if delta > 0:   # getting better
            if abs(delta) < self.min_delta:  # but not enough
                self.no_steps += 1
            else:   # really getting better
                self.no_steps = 0
                self.best_value = new_value
                if self.restore_best_weights:
                    self.best_model = trainer.model.state_dict()
        else:  # getting worse
            self.no_steps += 1

        return self._stop(trainer)

    def _init_params(self, trainer):
        values = trainer.history.get(self.monitor)
        if not values:
            raise ValueError(f"Monitor f{self.monitor} not found.")
        self.best_value = values[-1]

        if self.restore_best_weights:
            self.best_model = trainer.model.state_dict()

    def _stop(self, trainer):
        if self.no_steps >= self.patience:
            if self.restore_best_weights:
                trainer.model.load_state_dict(self.best_model)
            return True
        return False


class ModelCheckpoint:
    def __init__(self,
                 filepath,
                 monitor="val_loss",
                 mode="min",
                 save_best_only=True,
                 save_weights_only=True):
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.best_value = None

    def on_epoch_end(self, trainer):
        if not self.save_best_only:  # always saving
            self._save_model(trainer)
            return False

        # saving only best weights
        if not self.best_value:  # first time called
            self._init_params(trainer)
            return False

        new_value = trainer.history.get(self.monitor)[-1]
        if self.mode == "max":
            delta = new_value - self.best_value
        else:
            delta = self.best_value - new_value

        if delta > 0:   # getting better
            self._save_model(trainer)

        return False

    def _init_params(self, trainer):
        values = trainer.history.get(self.monitor)
        if not values:
            raise ValueError(f"Monitor f{self.monitor} not found.")
        self.best_value = values[-1]

        self._save_model(trainer)

    def _parse_filepath(self, trainer):
        namespace = {'epoch': len(trainer.history.get('train_loss'))}
        for key, value in trainer.history.items():
            namespace[key] = value[-1]

        return self.filepath.format(**namespace)

    def _save_model(self, trainer):
        parsed_filepath = self._parse_filepath(trainer)
        if self.save_weights_only:
            torch_save(trainer.model.state_dict(), parsed_filepath)
        else:
            torch_save(trainer.model, parsed_filepath)

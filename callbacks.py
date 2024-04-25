from tensorflow.keras.callbacks import Callback
import numpy as np


class MASECallback(Callback):
    def __init__(self, y_train, y_val, X_train, X_val):
        super().__init__()
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        self.mae_naive_train = np.mean(np.abs(y_train[1:] - y_train[:-1]))
        self.mae_naive_val = np.mean(np.abs(y_val[1:] - y_val[:-1]))

    def on_epoch_end(self, epoch, logs=None):
        y_train_pred = self.model.predict(self.X_train)
        y_val_pred = self.model.predict(self.X_val)

        mae_model_train = np.mean(np.abs(self.y_train - y_train_pred))
        mae_model_val = np.mean(np.abs(self.y_val - y_val_pred))

        mase_train = mae_model_train / self.mae_naive_train
        mase_val = mae_model_val / self.mae_naive_val

        print(f"Epoch {epoch + 1}: MASE train = {mase_train:.4f}, MASE val = {mase_val:.4f}")
        logs['mase_train'] = mase_train
        logs['mase_val'] = mase_val

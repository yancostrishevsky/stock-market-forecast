from sklearn.metrics import mean_squared_error
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, MultiHeadAttention, LayerNormalization, Embedding, Input, \
    GlobalAveragePooling1D
from tensorflow.keras import Model


def create_model(input_shape, output_size = 1):
    model = keras.models.Sequential([
        keras.layers.SimpleRNN(50, return_sequences=True, input_shape=input_shape),
        keras.layers.SimpleRNN(50),
        keras.layers.Dense(50, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(output_size)
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def create_lstm_model(input_shape, output_size=1):
    model = keras.models.Sequential([
        keras.layers.LSTM(100, return_sequences=True, input_shape=input_shape),
        keras.layers.LSTM(100),
        keras.layers.Dense(50, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(output_size)
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def create_gru_model(input_shape, output_size=1):
    model = keras.models.Sequential([
        keras.layers.GRU(100, return_sequences=True, input_shape=input_shape),
        keras.layers.GRU(100),
        keras.layers.Dense(50, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(output_size)
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def create_lstm_cnn_model(input_shape, output_size=1):
    model = keras.models.Sequential([
        keras.layers.Conv1D(filters=128, kernel_size=5, strides=1, padding="causal", activation="relu", input_shape=input_shape),
        keras.layers.BatchNormalization(),
        keras.layers.LSTM(100, return_sequences=True),
        keras.layers.Dropout(0.3),
        keras.layers.LSTM(100),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(50, activation='relu'),
        keras.layers.Dense(output_size)
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def create_cnn_lstm_model(input_shape, output_size=1):
    model = Sequential([
        keras.layers.Conv1D(filters=128, kernel_size=5, strides=1, padding="causal", activation="relu", input_shape=input_shape),
        keras.layers.MaxPooling1D(pool_size=2),
        keras.layers.Conv1D(filters=64, kernel_size=3, strides=1, padding="causal", activation="relu"),
        keras.layers.LSTM(100, return_sequences=True),
        keras.layers.LSTM(100),
        keras.layers.Dense(50, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(output_size)
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')
    return model


def create_bilstm_model(input_shape, output_size=1):
    model = Sequential([
        keras.layers.Bidirectional(keras.layers.LSTM(100, return_sequences=True), input_shape=input_shape),
        keras.layers.Bidirectional(keras.layers.LSTM(100)),
        keras.layers.Dense(50, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(output_size)
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')
    return model


def create_gru_dropout_model(input_shape, output_size=1):
    model = Sequential([
        keras.layers.GRU(100, return_sequences=True, input_shape=input_shape),
        keras.layers.Dropout(0.3),
        keras.layers.GRU(100),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(50, activation='relu'),
        keras.layers.Dense(output_size)
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')
    return model


def res_block(x, filters, kernel_size):
    shortcut = x
    x = keras.layers.Conv1D(filters, kernel_size, padding='same', activation='relu')(x)
    x = keras.layers.Conv1D(filters, kernel_size, padding='same')(x)

    # Dopasowanie wymiaru shortcut
    shortcut = keras.layers.Conv1D(filters, 1, padding='same')(shortcut)

    x = keras.layers.Add()([x, shortcut])
    x = keras.layers.Activation('relu')(x)
    return x


def create_resnet_model(input_shape, output_size=1):
    inputs = keras.Input(shape=input_shape)
    x = keras.layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(inputs)
    x = res_block(x, 64, 3)
    x = keras.layers.MaxPooling1D(pool_size=2)(x)

    x = res_block(x, 128, 3)
    x = keras.layers.MaxPooling1D(pool_size=2)(x)

    x = res_block(x, 256, 3)
    x = keras.layers.GlobalAveragePooling1D()(x)

    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dropout(0.3)(x)

    outputs = keras.layers.Dense(output_size)(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')
    return model


def create_attention_model(input_shape, output_size=1):
    inputs = keras.Input(shape=input_shape)
    x = keras.layers.LSTM(100, return_sequences=True)(inputs)
    x = keras.layers.LSTM(100, return_sequences=True)(x)

    attention = keras.layers.Dense(1, activation='tanh')(x)
    attention = keras.layers.Flatten()(attention)
    attention = keras.layers.Activation('softmax')(attention)
    attention = keras.layers.RepeatVector(100)(attention)
    attention = keras.layers.Permute([2, 1])(attention)

    x = keras.layers.multiply([x, attention])
    x = keras.layers.LSTM(100)(x)

    x = keras.layers.Dense(50, activation='relu')(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(output_size)(x)

    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')
    return model


def create_transformer_model(input_shape, output_size=1, head_size=64, num_heads=4, ff_dim=4, num_transformer_blocks=3,
    mlp_units=[128], dropout=0.2, mlp_dropout=0.2):
    inputs = Input(shape=input_shape)


    x = inputs

    for _ in range(num_transformer_blocks):
        attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(x, x)
        attention_output = Dropout(dropout)(attention_output)
        x = LayerNormalization(epsilon=1e-6)(x + attention_output)


        ff_output = Dense(ff_dim, activation="relu")(x)
        ff_output = Dropout(dropout)(ff_output)
        x = LayerNormalization(epsilon=1e-6)(x + ff_output)

    x = GlobalAveragePooling1D(data_format="channels_first")(x)


    for units in mlp_units:
        x = Dense(units, activation="relu")(x)
        x = Dropout(mlp_dropout)(x)

    outputs = Dense(output_size)(x)

    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def last_time_step_mse(Y_true, Y_pred):
    return keras.metrics.mean_squared_error(Y_true[:, -1], Y_pred[:, -1])


# model.py
def train_model(model, X_train, y_train, X_valid, y_valid, callbacks=None):
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,
        verbose=1,
        restore_best_weights=True
    )

    if callbacks:
        callbacks.append(early_stopping)
    else:
        callbacks = [early_stopping]

    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_valid, y_valid),
        callbacks=callbacks,
        verbose=1
    )
    return history



def save_model(model, file_name):
    model.save(file_name)

def load_model(file_name):
    return keras.models.load_model(file_name)

import numpy as np

def naive_forecasting_mae(y_train):
    """Oblicza MAE używając naivnego modelu prognozującego dla danych treningowych."""
    return np.mean(np.abs(y_train[1:] - y_train[:-1]))

def mase(y_true, y_pred, y_train):
    """Oblicza Mean Absolute Scaled Error (MASE)."""
    mae_model = np.mean(np.abs(y_true - y_pred))
    mae_naive = naive_forecasting_mae(y_train)
    return mae_model / mae_naive

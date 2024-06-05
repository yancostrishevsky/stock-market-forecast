import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

def load_and_scale_data(file_path):
    data = pd.read_csv(file_path)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data[['Zamkniecie']])
    return data_scaled, scaler

def create_dataset(data, n_steps, future_steps=1):
    X, y = [], []
    for i in range(n_steps, len(data) - future_steps + 1):
        X.append(data[i - n_steps:i, 0])
        y_seq = data[i:i + future_steps, 0]
        y.append(y_seq)
    return np.array(X), np.array(y)

def plot_predictions(real_time_series, predicted_futures, real_futures, n_steps):
    plt.figure(figsize=(12, 8))

    colors = cycle(['black', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray'])
    model_labels = ['Attention', 'BiLSTM', 'GRU', 'GRU Dropout', 'LSTM', 'LSTM CNN', 'ResNet', 'RNN']


    plt.plot(range(0, len(real_futures)), real_futures, label='Real Future', marker='.', color='red', zorder=-1)

    for predicted_future, color, label in zip(predicted_futures, colors, model_labels):
        plt.plot(range(0, len(predicted_future)), predicted_future, color=color, marker='.', zorder=2, label=label)

    plt.title('Porównanie rzeczywistej serii czasowej z przewidywaniami modeli')
    plt.xlabel('Czas [dni]')
    plt.ylabel('wartość [PLN]')
    plt.legend()
    plt.grid(True)
    plt.show()

file_path = 'wig20_d.csv'
data_scaled, scaler = load_and_scale_data(file_path)

n_steps = 90
future_steps = 1

X, y = create_dataset(data_scaled, n_steps, future_steps=future_steps)

split_point1 = int(0.7 * len(X))
split_point2 = int(0.9 * len(X))
X_train, y_train = X[:split_point1], y[:split_point1]
X_valid, y_valid = X[split_point1:split_point2], y[split_point1:split_point2]
X_test, y_test = X[split_point2:], y[split_point2:]

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_valid = X_valid.reshape((X_valid.shape[0], X_valid.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

paths = [
    'models/model_attention_1.keras',
    'models/model_bilstm_1.keras',
    'models/model_gru_1.keras',
    'models/model_gru_dropout_1.keras',
    'models/model_lstm_1.keras',
    'models/model_lstm_cnn_1.keras',
    'models/model_resnet_1.keras',
    'models/model_rnn_1.keras'
]

predicts = []

for mod in paths:
    model = load_model(mod)
    predicted_futures = []
    for start in range(150, 210):
        X_input = X_test[start]
        y_pred = model.predict(X_input.reshape(1, n_steps, 1))
        y_pred_rescaled = scaler.inverse_transform(y_pred).flatten()
        predicted_futures.extend(y_pred_rescaled[:future_steps])
    predicts.append(predicted_futures[:60])

y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, y_test.shape[-1]))

test_index = 175
real_time_series = scaler.inverse_transform(X_test[test_index].reshape(-1, 1)).flatten()
real_futures = y_test_rescaled[149:149 + 60].flatten()

print(len(real_time_series), len(predicts[0]), len(real_futures))
plot_predictions(real_time_series, predicts, real_futures, 30)

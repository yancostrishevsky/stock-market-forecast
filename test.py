from data_preparation import load_and_scale_data, create_dataset
from model import create_lstm_model, train_model, save_model, create_lstm_cnn_model, create_gru_model
from plotting import plot_predictions
from model import load_model
from model import mase
file_path = 'wig20_d.csv'  # Przykładowa ścieżka do pliku z danymi
data_scaled, scaler = load_and_scale_data(file_path)

n_steps = 100  # Liczba kroków wstecz, jakie chcemy uwzględnić
future_steps = 10  # Kroków do przodu, które chcemy przewidzieć

X, y = create_dataset(data_scaled, n_steps, future_steps=future_steps)

# Dzielimy dane na zestawy: treningowy, walidacyjny i testowy
split_point1 = int(0.7 * len(X))
split_point2 = int(0.9 * len(X))
X_train, y_train = X[:split_point1], y[:split_point1]
X_valid, y_valid = X[split_point1:split_point2], y[split_point1:split_point2]
X_test, y_test = X[split_point2:], y[split_point2:]

# Dostosowujemy kształt danych do wymagań modelu LSTM
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_valid = X_valid.reshape((X_valid.shape[0], X_valid.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))



input_shape = (n_steps, 1)  # Kształt danych wejściowych

path = 'model_trans.keras'
model = load_model(path)



y_pred = model.predict(X_test)
y_pred_rescaled = scaler.inverse_transform(y_pred)
y_test_rescaled = scaler.inverse_transform(y_test)


# Przykładowy indeks w danych testowych do wizualizacji
test_index = -1
real_time_series = scaler.inverse_transform(X_test[test_index].reshape(-1, 1)).flatten()
predicted_futures = y_pred_rescaled[test_index]
real_futures = y_test_rescaled[test_index]

mase_value = mase(y_test_rescaled.flatten(), y_pred_rescaled.flatten(), y_train)
print(f"MASE: {mase_value}")

plot_predictions(real_time_series, predicted_futures, real_futures, n_steps)

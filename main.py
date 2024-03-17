from data_preparation import load_and_scale_data, create_dataset
from model import create_lstm_model, train_model, save_model
from plotting import plot_predictions

file_path = 'wig20_d.csv'
n_steps = 90

data_scaled, scaler = load_and_scale_data(file_path)
X, y = create_dataset(data_scaled, n_steps)

split_point1 = int(0.7 * len(X))
split_point2 = int(0.9 * len(X))
X_train, y_train = X[:split_point1], y[:split_point1]
X_valid, y_valid = X[split_point1:split_point2], y[split_point1:split_point2]
X_test, y_test = X[split_point2:], y[split_point2:]
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_valid = X_valid.reshape((X_valid.shape[0], X_valid.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))


model = create_lstm_model((n_steps, 1))
history = train_model(model, X_train, y_train, X_valid, y_valid)

save_model(model, 'model_lstm.h5')


y_pred = model.predict(X_test)

y_pred_rescaled = scaler.inverse_transform(y_pred)
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# ostatnia seria czasową z zestawu testowego i jej przewidywanie
time_series_real_rescaled = y_test_rescaled[-1]
time_series_pred_rescaled = y_pred_rescaled[-1]

real_time_series = scaler.inverse_transform(X_test[-1].reshape(-1, 1)).flatten()

real_future_value = y_test_rescaled[-1]

predicted_future_value = y_pred_rescaled[-1]

plot_predictions(
    real_time_series=real_time_series,
    predicted_future=predicted_future_value,
    real_future=real_future_value,
    n_steps=n_steps
)

#  do przewidzenia kolejnego dnia
next_day_prediction = model.predict(X_test[-1].reshape((1, n_steps, 1)))
next_day_prediction_rescaled = scaler.inverse_transform(next_day_prediction)
print(f"Przewidywana wartość na jutro: {next_day_prediction_rescaled[0][0]}")

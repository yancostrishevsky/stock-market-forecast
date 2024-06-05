from callbacks import MASECallback
from data_preparation import load_and_scale_data, create_dataset
from model import create_lstm_model, train_model, save_model, create_lstm_cnn_model, create_gru_model, create_transformer_model, create_cnn_lstm_model, create_bilstm_model, create_gru_dropout_model, create_resnet_model, create_attention_model, create_model
from plotting import plot_predictions


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



input_shape = (n_steps, 1)

mase_callback = MASECallback(y_train, y_valid, X_train, X_valid)
model = create_attention_model(input_shape, output_size=future_steps)
history = train_model(model, X_train, y_train, X_valid, y_valid, callbacks=[mase_callback])

save_model(model, 'models/model_bilstm_1.keras')


y_pred = model.predict(X_test)
y_pred_rescaled = scaler.inverse_transform(y_pred)
y_test_rescaled = scaler.inverse_transform(y_test)



test_index = -1
real_time_series = scaler.inverse_transform(X_test[test_index].reshape(-1, 1)).flatten()
predicted_futures = y_pred_rescaled[test_index]
real_futures = y_test_rescaled[test_index]

plot_predictions(real_time_series, predicted_futures, real_futures, n_steps)

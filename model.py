from tensorflow import keras

def create_model(input_shape):
    model = keras.models.Sequential([
        keras.layers.SimpleRNN(20, return_sequences=True, input_shape=input_shape),
        keras.layers.SimpleRNN(20),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def create_lstm_model(input_shape):
    model = keras.models.Sequential([
        keras.layers.LSTM(50, return_sequences=True, input_shape=input_shape),
        keras.layers.LSTM(50),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model



def train_model(model, X_train, y_train, X_valid, y_valid):
    history = model.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid))
    return history

def save_model(model, file_name):
    model.save(file_name)

def load_model(file_name):
    return keras.models.load_model(file_name)

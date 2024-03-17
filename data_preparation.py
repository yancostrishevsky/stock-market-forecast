import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def load_and_scale_data(file_path):
    data = pd.read_csv(file_path)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data[['Otwarcie']])
    return data_scaled, scaler

def create_dataset(data, n_steps):
    X, y = [], []
    for i in range(n_steps, len(data)):
        X.append(data[i-n_steps:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

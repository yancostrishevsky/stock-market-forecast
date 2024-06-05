import matplotlib.pyplot as plt
from itertools import cycle

def plot_predictions(real_time_series, predicted_futures, real_futures, n_steps):
    plt.figure(figsize=(10, 6))

    extended_time_series_x = list(range(n_steps)) + [n_steps + i for i in range(len(real_futures))]

    extended_time_series_y = list(real_time_series[-30:]) + list(real_futures)

    plt.plot(extended_time_series_x, extended_time_series_y, label='Real Time Series + Future', marker='.', zorder=-1)

    colors = cycle(['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray'])
    model_labels = ['Attention', 'BiLSTM', 'GRU', 'GRU Dropout', 'LSTM', 'LSTM CNN', 'ResNet', 'RNN']

    for i, (predicted_future, color, label) in enumerate(zip(predicted_futures, colors, model_labels)):
        plt.plot(range(n_steps, n_steps + len(predicted_future)), predicted_future, color=color, marker='.', zorder=2, label=label)

    plt.title('Comparison of Real and Predicted Time Series Values')
    plt.xlabel('Time [days]')
    plt.ylabel('Value [PLN]')
    plt.legend()
    plt.grid(True)
    plt.show()


import matplotlib.pyplot as plt

def plot_predictions(real_time_series, predicted_future, real_future, n_steps):
    plt.figure(figsize=(10, 6))
    plt.plot(range(n_steps), real_time_series, label='Rzeczywista seria czasowa', marker='.', zorder=-1)
    plt.scatter(n_steps, real_future, color='orange', label='Rzeczywista wartość przyszła', zorder=1)
    plt.scatter(n_steps, predicted_future, color='red', label='Przewidywana wartość przyszła', zorder=2)
    plt.title('Porównanie rzeczywistych i przewidywanych wartości serii czasowej')
    plt.xlabel('Czas')
    plt.ylabel('Wartość')
    plt.legend()
    plt.grid(True)
    plt.show()

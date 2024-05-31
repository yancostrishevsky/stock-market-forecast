import matplotlib.pyplot as plt


def plot_predictions(real_time_series, predicted_futures, real_futures, n_steps):
    plt.figure(figsize=(10, 6))

    # Tworzymy rozszerzoną oś X dla połączenia serii czasowej i przyszłych wartości
    extended_time_series_x = list(range(n_steps)) + [n_steps + i for i in range(len(real_futures))]

    # Rozszerzamy rzeczywistą serię czasową o rzeczywiste przyszłe wartości
    extended_time_series_y = list(real_time_series) + list(real_futures)

    # Rysujemy rozszerzoną rzeczywistą serię czasową
    plt.plot(extended_time_series_x, extended_time_series_y, label='Rzeczywista seria czasowa + przyszłość', marker='.',
             zorder=-1)

    if len(predicted_futures) > 1:
        # Dla wielu przewidywanych wartości rysujemy linię
        plt.plot(range(n_steps, n_steps + len(predicted_futures)), predicted_futures, color='red',
                 label='Przewidywane wartości przyszłe', zorder=2)
    else:
        # Dla jednej przewidywanej wartości używamy punktu
        plt.scatter(n_steps, predicted_futures[0], color='red', label='Przewidywana wartość przyszła', zorder=2)

    plt.title('Porównanie rzeczywistych i przewidywanych wartości serii czasowej')
    plt.xlabel('Czas')
    plt.ylabel('Wartość')
    plt.legend()
    plt.grid(True)
    plt.show()

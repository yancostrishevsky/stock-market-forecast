# Raport - Przewidywanie kursów akcji na podstawie danych historycznych. Analiza porównawcza modeli sieci neuronowych.

## Jan Stryszewski, Jakub Truszkowski



# Modyfikacja Procesu Przygotowania Danych

W ramach naszego projektu analizy porównawczej różnych modeli sieci neuronowych zdecydowaliśmy się na wprowadzenie istotnych modyfikacji w kodzie służącym do przygotowania danych. Głównym celem tych zmian było umożliwienie łatwego dostosowywania wymiarów danych wejściowych do modelu, w zależności od ilości kroków czasowych (`n_steps`), które chcemy uwzględnić w analizie, oraz liczby przewidywanych kroków (`future_steps`). Dzięki temu jesteśmy w stanie w elastyczny sposób testować i porównywać wydajność modeli przy różnych ustawieniach horyzontu predykcji.

## Funkcja `load_and_scale_data`

Pierwszą modyfikacją jest funkcja `load_and_scale_data`, która wczytuje dane z pliku CSV i stosuje na nich normalizację za pomocą `MinMaxScaler`. 

```python
def load_and_scale_data(file_path):
    data = pd.read_csv(file_path)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data[['Zamkniecie']])
    return data_scaled, scaler
```


Kolejną zmianą jest funkcja `create_dataset`, która przekształca przeskalowane dane w sekwencje wejściowe i odpowiadające im etykiety (wartości docelowe). Ta funkcja umożliwia dynamiczne dostosowanie ilości kroków (`n_steps`) wykorzystanych do generowania sekwencji wejściowych oraz liczby kroków do przodu (`future_steps`), które model ma przewidzieć.

Parametry:
- `data` (numpy array): Przeskalowane dane, z których tworzone są sekwencje.
- `n_steps` (int): Liczba kroków wstecz, które mają być uwzględnione w każdej sekwencji.
- `future_steps` (int): Liczba kroków do przodu, które chcemy przewidzieć. Domyślnie ustawiona na 1.

### Zwraca:
- `X` (numpy array): Tablica sekwencji wejściowych do modelu.
- `y` (numpy array): Tablica etykiet odpowiadających sekwencjom wejściowym.

### Szczegóły implementacji:


```python
def create_dataset(data, n_steps, future_steps=1):
    X, y = [], []
    for i in range(n_steps, len(data) - future_steps + 1):
        X.append(data[i-n_steps:i, 0])
        y_seq = data[i:i+future_steps, 0]
        y.append(y_seq)
    return np.array(X), np.array(y)
```
Dzięki wprowadzeniu tych modyfikacji, nasz proces przygotowania danych stał się bardziej elastyczny i przystosowany do kompleksowej analizy porównawczej modeli.

# Prototypy Modeli

Na dzień 19.04.2024 w projekcie utworzyliśmy zadeklarowane architektury modeli. Każdy z modeli został juz skompilowany aby sprawdzić jego działanie, narazie są to prototypowe wersje. W kazdym zaimplementowaliśmy parametr pozwalający na szybką zmianę rozmiaru wyjścia, gdyz chcemy analizować ich skuteczność w kilku wariantach kroków czasowych.

## Simple RNN Model

```python
def create_model(input_shape):
    model = keras.models.Sequential([
        keras.layers.SimpleRNN(20, return_sequences=True, input_shape=input_shape),
        keras.layers.SimpleRNN(20),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
```
Model składa się z dwóch warstw `SimpleRNN`, z których każda zawiera 20 jednostek. `return_sequences=True` w pierwszej warstwie umożliwia przekazanie sekwencji do kolejnej warstwy RNN. Ostatecznie, warstwa `Dense` agreguje wyniki do pojedynczej prognozy na wyjściu. Model ten służy jako punkt odniesienia dla bardziej złożonych architektur.

## Model z komórkami pamięci LSTM
```python

def create_lstm_model(input_shape, output_size=5):
    model = keras.models.Sequential([
        keras.layers.LSTM(50, return_sequences=True, input_shape=input_shape),
        keras.layers.LSTM(50),
        keras.layers.Dense(output_size)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

```
Model wykorzystuje dwie warstwy `LSTM` z 50 jednostkami, zapewniając lepsze zachowanie długoterminowych zależności w danych dzięki mechanizmom `bramek LSTM`. Wynik jest przekazywany do warstwy `Dense`, która generuje prognozy o rozmiarze określonym przez `output_size`. 

## Model z komórkami pamięci GRU
```python
def create_gru_model(input_shape):
    model = keras.models.Sequential([
        keras.layers.GRU(50, return_sequences=True, input_shape=input_shape),
        keras.layers.GRU(50),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

```
Podobnie jak model `LSTM`, ten model wykorzystuje dwie warstwy `GRU`, które są uznawane za uproszczoną wersję `LSTM`. Są one jednak równie efektywne i często szybsze w trenowaniu.

## Model sieci konwolucyjnej z komórkami LSTM
```python
def create_lstm_cnn_model(input_shape, output_size=1):
    model = keras.models.Sequential([
        keras.layers.Conv1D(filters=64, kernel_size=5, strides=1, padding="causal", activation="relu", input_shape=input_shape),
        keras.layers.BatchNormalization(),
        keras.layers.LSTM(50, return_sequences=True),
        keras.layers.Dropout(0.2),
        keras.layers.LSTM(50),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(output_size)
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

```
Ta hybrydowa architektura łączy warstwy konwolucyjne (`Conv1D`) z `LSTM`, wykorzystując konwolucje do efektywnego przetwarzania sekwencji czasowych i ekstrakcji cech, a `LSTM` do modelowania zależności sekwencyjnych. `BatchNormalization` i `Dropout` są używane do poprawy stabilności i generalizacji modelu. Do wykorzystania takiego rozwiązania inspirowaliśmy się architekturą sieci Wavenet, która w bardzo duzym uproszczeniu przypomina takie połączenie.

## Model Transformerowy
```python
def create_transformer_model(input_shape, output_size=1, head_size=64, num_heads=4, ff_dim=4, num_transformer_blocks=3, mlp_units=[128], dropout=0.2, mlp_dropout=0.2):
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

```
W prototypowym modelu skonfigurowaliśmy podstawowe bloki transformera, warstwy uwagi (`MultiHeadAttention`), z normalizacją warstw (`LayerNormalization`) i prostymi sieciami `feed-forward` w każdym bloku transformera. Liczba bloków transformera, rozmiar uwagi i inne hiperparametry bedziemy dostosowywać w trakcie uczenia.
Na końcu używamy `GlobalAveragePooling1D` i jednej lub więcej gęstych warstw (`Dense`) do przetworzenia wyników na przewidywane wartości wyjściowe.

# Czym zajmujemy się na ten moment
Po wstępnym przetestowaniu naszych architektur napotkaliśmy problem z dopasowaniem dobrych metryk do oszacowania błędu pomiędzy rzeczywistymi a przewidywanymi wartosciami przebiegów czasowych. Do tej pory wykorzystywaliśmy MSE(błąd średniokwadratowy) i mimo, ze otrzymywaliśmy bardzo małe wartości błędu, to rozbiezności na wykresach były duze. Wynikało to z tego, ze nasze dane są przeskalowane do wartości [0,1].

Postanowiliśmy zaimplementować metrykę Mean Absolute Scaled Error. 
# Mean Absolute Scaled Error (MASE)

MASE to metryka wydajności modelu prognozującego, która normalizuje średni błąd bezwzględny (Mean Absolute Error - MAE) przez średni błąd bezwzględny uzyskany z prostego modelu naivnego. W kontekście serii czasowych, naivny model prognozujący często przyjmuje, że najlepszą prognozą na następny krok czasowy jest wartość z poprzedniego kroku.

## Definicja

MASE jest zdefiniowane jako:

$$
MASE = \frac{MAE}{MAE_{naiv}}
$$

gdzie:

- $MAE$ to średni błąd bezwzględny modelu, który jest oceniany,
- $MAE_{naiv}$ to średni błąd bezwzględny prostego modelu naivnego.

## Obliczanie $MAE_{naiv}$

Dla serii czasowej $y_t$ (gdzie $t$ oznacza indeks czasowy), $MAE_{naiv}$ jest zwykle obliczane jako średnia wartość bezwzględnych różnic między kolejnymi obserwacjami:

$$
MAE_{naiv} = \frac{1}{T-1} \sum_{t=2}^{T} |y_t - y_{t-1}|
$$

gdzie $T$ to całkowita liczba obserwacji.

## Interpretacja

Wartość MASE mniejsza niż 1 oznacza, że model prognozujący ma mniejszy błąd niż prosty model naivny. Z kolei wartość MASE większa niż 1 wskazuje, że model jest gorszy od prostego przewidywania wartości z poprzedniego kroku. 

MASE ma kilka kluczowych zalet:

- **Niezmienność skali:** Dzięki normalizacji, MASE może być stosowane do porównywania modeli na różnych serii czasowych, niezależnie od ich skali.
- **Łatwość interpretacji:** Wartość MASE bezpośrednio mówi, jak model wypada w porównaniu do modelu naivnego.
- **Odporność na anomalie:** W przeciwieństwie do innych metryk, takich jak RMSE, MASE jest mniej wrażliwe na duże odchylenia lub anomalie w danych.


## Implementacja metryki MASE
Bezpośrednie używanie MASE jako metryki w procesie trenowania modelu w Kerasie jest trudne ze względu na specyficzną naturę tej metryki, która wymaga dostępu do danych treningowych w celu obliczenia średniego błędu bezwzględnego (MAE) dla naiwnego modelu prognozującego. Ponieważ Keras ogranicza metryki do funkcji przyjmujących jedynie przewidywane wartości i prawdziwe etykiety jako argumenty, bezpośrednie włączenie MASE jako metryki trenowania jest niepraktyczne. Dlatego utworzyliśmy callback Kerasa, który oblicza MASE po każdej epoce treningu i używa go do monitorowania wydajności modelu. 

```python
def naive_forecasting_mae(y_train):
    return np.mean(np.abs(y_train[1:] - y_train[:-1]))

def mase(y_true, y_pred, y_train):
    mae_model = np.mean(np.abs(y_true - y_pred))
    mae_naive = naive_forecasting_mae(y_train)
    return mae_model / mae_naive
```
Funkcja `naive_forecasting_mae`
oblicza MAE używając naiwnego modelu prognozującego dla danych treningowych. 

```python
class MASECallback(Callback):
    def __init__(self, y_train, y_valid, scaler):
        super(MASECallback, self).__init__()
        self.y_train = y_train
        self.y_valid = y_valid
        self.scaler = scaler

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.validation_data[0])
        y_pred_rescaled = self.scaler.inverse_transform(y_pred)
        y_valid_rescaled = self.scaler.inverse_transform(self.y_valid.reshape(-1, 1))
        
        mase_value = mase(y_valid_rescaled.flatten(), y_pred_rescaled.flatten(), self.y_train)
        print(f"\nEpoch {epoch+1} MASE: {mase_value:.4f}")

```
Zaimplementowaliśmy klasę callback MASE i tworzymy jej instancję w funkcji `train_model` razem z `EarlyStopping`,  a następnie dodajemy ją do listy callbacków w funkcji `model.fit()`. 
Mechanizm Early Stopping to technika stosowana podczas trenowania modeli sieci neuronowych, która pozwala zakończyć proces uczenia, gdy model przestaje się poprawiać na zbiorze walidacyjnym. 

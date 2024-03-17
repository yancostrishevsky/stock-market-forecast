# Time Series Forecasting with RNN's

This project is focused on forecasting time series data using various types of RNN's like Long Short-Term Memory (LSTM) neural networks. It demonstrates the process of data preparation, model training, and prediction using Python. Specifically, we utilize stock market data (e.g., WIG20 index) to predict future values based on past performance.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

- Python 3.x
- Pip (Python package installer)

### Installation

1. **Clone the repository**
  ```bash
  git clone https://github.com/yourusername/your-repository-name.git
  ```
2. **Navigate to the project directory**
   ```bash
   cd your-repository-name
   ```
3. **Create a virtual environment**
   ```bash
   python -m venv venv
   ```
4. **Activate the virtual environment**
   On Windows:
   ```bash
   venv\Scripts\activate
   ```
   On Mac/linux:
   ```bash
   source venv/bin/activate
   ```
5. **Install requirements.txt**
   ```bash
   pip install -r requirements.txt
    ```
### USAGE
- data_preparation.py: Contains functions for loading and preparing the time series data.
- model.py: Includes the definition of the LSTM model, training, and saving/loading mechanisms.
- plotting.py: Provides functionalities to visualize the real vs. predicted data.
- main.py: The main script that orchestrates the data preparation, model training, and plotting.

### Contributing

Please read CONTRIBUTING.md for details on our code of conduct, and the process for submitting pull requests to us.

### License

This project is licensed under the MIT License - see the LICENSE.md file for details

### Acknowledgments

- Hat tip to anyone whose code was used
- Inspiration: nio nio

   




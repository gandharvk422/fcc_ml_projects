# Stock Price Prediction using Time Series Analysis

This project aims to predict stock prices using Long Short-Term Memory (LSTM) networks, a type of recurrent neural network (RNN), implemented in Python with Keras and TensorFlow.

## Overview

The provided code utilizes time series analysis techniques and LSTM networks to forecast stock prices based on historical data. Here's a brief overview of the project structure and functionality:

1. **Data Preprocessing**: The historical stock price data is preprocessed, including scaling the data and structuring it into sequences suitable for training the LSTM model.

2. **Building and Training the LSTM Model**: An LSTM-based recurrent neural network is constructed and trained on the preprocessed data. The model architecture includes multiple LSTM layers with dropout regularization to prevent overfitting.

3. **Making Predictions**: After training, the model is used to make predictions on unseen data (future stock prices). The predicted stock prices are then inverse transformed to obtain the actual stock price values.

4. **Visualizing Results**: The actual and predicted stock prices are visualized using matplotlib to evaluate the performance of the model and analyze the accuracy of predictions.

## Prerequisites

Before running the code, ensure you have the following dependencies installed:

- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Keras
- TensorFlow

You can install these dependencies via pip:


## Usage

1. Clone the repository or download the project files.

2. Place your historical stock price data in CSV format in the `dataset` directory. Ensure the file names match the ones specified in the code (`Google_Stock_Price_Train.csv` and `Google_Stock_Price_Test.csv`).

3. Run the Python script (`stock_price_prediction.py`) to execute the project code. This will train the LSTM model, make predictions, and visualize the results.

## Notes

- Adjust hyperparameters, such as the number of LSTM layers, dropout rates, epochs, and batch size, as needed to improve model performance.
- Experiment with different feature engineering techniques and alternative machine learning algorithms for comparison.
- Keep in mind that stock price prediction is inherently uncertain, and predictions should be interpreted with caution.

## License

This project is licensed under the [MIT License](LICENSE).
<hr>
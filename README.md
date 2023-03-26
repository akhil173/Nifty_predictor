# Nifty_predictor
A Stacked LSTM model, that is trained on Nifty hourly data and predicts the future Nifty Index value based on Close data inputs from 150 previous hourly candles.



# Data Resampling
Pandas Library is used to resample the minute wise candles of Nifty Data to Hourly Data.
The Hourly Nifty data is saved in .parquet format to efficiently store the processed data and decrease the space required.

The nifty hourly data parquet file is attached for reference and the data is of the format of ["Datetime", "Ticker", "Open, "High", "Low", "Close", "Volume"].


# Model Training
The Model is basically a stacked LSTM, i.e, we initialise the model as a Sequential Model and then stack 3 LSTM models to it.

Also since LSTM models are sensitive of the scaling of values, we use MinMaxScaler from sklearn.preprocessing to scale the Close values of Nifty to range between [0, 1].
The loss of this model is set to mean_squared_error and optimizer is set as 'adam' optimizer.

We then train the model with epoch=150 ( The epoch value can be variable according to one's intuition ).


The model is then saved using TensorFlow's SavedModel format.


# Required Libraries
tensorflow\n
keras\n
pandas\n
numpy\n
sklearn\n
matplotlib\n
tvDatafeed\n


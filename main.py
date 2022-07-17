import sys
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

def plot_predictions(model, X, y, start=0, end=100):
    predictions = model.predict(X).flatten()
    df = pd.DataFrame(data={'Predictions': predictions, 'Actuals': y})
    plt.plot(df['Predictions'][start:end], '--', label = 'predicted with model 3')
    plt.plot(df['Actuals'][start:end], '-', label = 'actual')
    plt.legend(fontsize=12)

def df_to_X_y(df, window_size, num_inputs):
    df_as_np = df.to_numpy()
    X = []
    y = []
    for i in range(len(df_as_np)-window_size):
        row = [r for r in df_as_np[i:i+window_size]]
        X.append(row)
        label = df_as_np[i+window_size][num_inputs-1]
        y.append(label)
    return np.array(X), np.array(y)



def main(argv):
    learning_rate = 1
    epochsize = 10
    window_size = 72

    parser = argparse.ArgumentParser(description='Prototype for water demand forecasting')
    parser.add_argument('inputmode', type=int, help='choose the input that is used to predict')
    parser.add_argument('-w', type=int, help='the window size that is used to predict')
    parser.add_argument('-e', type=int, help='the epoch size that is used to predict')
    parser.add_argument('-l', type=float, help='the learning rate that is used to predict')

    args = parser.parse_args()

    inputmode = args.inputmode
    if args.w is not None:
        window_size = args.w
    if args.e is not None:
        epochsize = args.e
    if args.l is not None:
        learning_rate = args.l

    df = pd.read_csv('data_formatted.csv')
    
    if inputmode == 1:
        num_inputs = 1
        flow_vol_df = pd.DataFrame({'flow_vol': df['flow_vol']})
    elif inputmode == 2:
        num_inputs = 3
        flow_vol_df = pd.DataFrame({'Weekday': df['Weekday'], 'Hour': df['Hour'], 'flow_vol': df['flow_vol']})
    else:
        num_inputs = 6
        flow_vol_df = pd.DataFrame({'Weekday': df['Weekday'], 'Hour': df['Hour'], 'Temperature': df['Temperature'], 'Humidity': df['Humidity'], 'Rain': df['Rain'], 'flow_vol': df['flow_vol']})
    
    X, y = df_to_X_y(flow_vol_df, window_size, num_inputs)

    n = len(df) - window_size

    X_train, y_train = X[:int(n*0.7)], y[:int(n*0.7)]
    X_val, y_val = X[int(n*0.7):1105], y[int(n*0.7):1105]
    X_test, y_test = X[1105 - window_size:], y[1105 - window_size:]

    model = Sequential()
    model.add(InputLayer((window_size, num_inputs)))
    model.add(LSTM(64))
    model.add(Dense(8, 'relu'))
    model.add(Dense(1, 'linear'))

    cp = ModelCheckpoint('model/', save_best_only=True)
    model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate), metrics=[RootMeanSquaredError()])

    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochsize, callbacks=[cp], verbose=2)

    model = tf.keras.models.load_model('model/')

    plot_predictions(model, X_test, y_test)

    plt.show()

if __name__ == "__main__":
    main(sys.argv[1:])
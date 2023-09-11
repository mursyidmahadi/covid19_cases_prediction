from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
import os, datetime, pickle

def eda(df):
    "Check for null values in target column"
    print(df.head())
    print(df.info())
    print(df['cases_new'].isna().sum())

def plot_data(df):
    "Plotting Time-Series Data"
    plt.figure()
    plt.plot(df['cases_new'].values)
    plt.ylabel("Number of new cases")
    plt.show()

def data_mms(WINDOW_SIZE, data):
    mms = MinMaxScaler()
    data = mms.fit_transform(np.expand_dims(data, axis=-1))

    X = []
    y = []

    for i in range(WINDOW_SIZE, len(data)):
        X.append(data[i-WINDOW_SIZE:i])
        y.append(data[i])

    X = np.array(X)
    y = np.array(y)

    return mms, X, y

def model_archi(X_train, MODEL_PNG_PATH):
    input_shape = np.shape(X_train)[1:]

    model = Sequential()
    model.add(LSTM(64, activation='tanh', input_shape=input_shape, return_sequences=True))
    model.add(LSTM(4, activation='tanh', input_shape=input_shape))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mae', metrics=['mape', 'mse'])

    model.summary()
    plot_model(model, to_file=MODEL_PNG_PATH, show_shapes=True)

    return model

def model_train(model, X_train, y_train, X_test, y_test):
    log_dir = os.path.join(os.getcwd(), 'tensorboard_logs', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tb_callback = TensorBoard(log_dir=log_dir)

    hist = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=64, epochs=600, callbacks=[tb_callback])
    return hist, model

def predict_score(data, prediction):
    print(f"MAE is {mean_absolute_error(data, prediction)}")
    print(f"MAPE is {mean_absolute_percentage_error(data, prediction)}")
    print(f"R2 value is {r2_score(data, prediction)}")

def test_prepare(TEST_PATH, WINDOW_SIZE, df, data, mms):
    df_test = pd.read_csv(TEST_PATH)
    df_total = pd.concat([df,df_test])
    df_total = df_total.reset_index(drop=True)
    df_total['cases_new'] = df_total['cases_new'].interpolate(method='polynomial', order=2)
    df_total = df_total['cases_new'].values

    df_total = mms.transform(np.expand_dims(df_total, axis=-1))

    X_actual = []
    y_actual = []

    for i in range(len(df), len(df_total)):
        X_actual.append(df_total[i-WINDOW_SIZE:i])
        y_actual.append(df_total[i])

    X_actual = np.array(X_actual)
    y_actual = np.array(y_actual)
    
    return X_actual, y_actual

def predict_plot(mms, y_actual, y_pred_actual):
    y_pred_actual_iv = mms.inverse_transform(y_pred_actual)
    y_actual_iv = mms.inverse_transform(y_actual)

    plt.figure()
    plt.plot(y_pred_actual_iv, color='red')
    plt.plot(y_actual_iv, color='blue')
    plt.legend(['Predicted cases', 'Actual cases'])
    plt.show()

def model_save(MODEL_PATH, PKL_PATH, model, mms):
    model.save(MODEL_PATH)

    with open(PKL_PATH, 'wb') as f:
        pickle.dump(mms, f)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
import os

def create_modal(AllOutPut, LSTM_MinMaxModel , Regression_X_train, Regression_y_train):
    KNNModel = KNeighborsRegressor(n_neighbors=5)
    KNNModel.fit(LSTM_MinMaxModel.transform(Regression_X_train), Regression_y_train)

    return KNNModel

def knn_regression_modal(NowDateTime, LSTM_MinMaxModel , AllOutPut, Regression_X_train, Regression_y_train):
    RegressionModel = create_modal(AllOutPut, LSTM_MinMaxModel , Regression_X_train, Regression_y_train)
    
    os.makedirs('./model', exist_ok=True)
    print( os.getcwd() )
    joblib.dump(RegressionModel, f'./model/WeatherRegression_{NowDateTime}')

    print('KNN R squared:', RegressionModel.score(LSTM_MinMaxModel.transform(Regression_X_train), Regression_y_train))
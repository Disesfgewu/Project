from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import MinMaxScaler
import joblib
import numpy as np
import pandas as pd
import os


def create_modal(AllOutPut, Regression_X_train, Regression_y_train):
    LSTM_MinMaxModel = MinMaxScaler().fit(AllOutPut)
    
    RegressionModel = ExtraTreesRegressor(n_estimators=100, random_state=42)
    RegressionModel.fit(LSTM_MinMaxModel.transform(Regression_X_train), Regression_y_train)
    
    return RegressionModel, LSTM_MinMaxModel

def ExtraTree_regression_modal(NowDateTime, AllOutPut, Regression_X_train, Regression_y_train):
    RegressionModel, LSTM_MinMaxModel = create_modal(AllOutPut, Regression_X_train, Regression_y_train)
    os.makedirs('./model', exist_ok=True)
    joblib.dump(RegressionModel, f'./model/WeatherRegression_{NowDateTime}')
    
    print('Feature Importances: ', RegressionModel.feature_importances_)
    print('R squared: ', RegressionModel.score(LSTM_MinMaxModel.transform(Regression_X_train), Regression_y_train))

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import joblib

import numpy as np
import pandas as pd
import os

def create_modal( AllOutPut , Regression_X_train , Regression_y_train ):
    LSTM_MinMaxModel = MinMaxScaler().fit(AllOutPut)
    RegressionModel = LinearRegression()
    RegressionModel.fit(LSTM_MinMaxModel.transform(Regression_X_train), Regression_y_train)
    return RegressionModel , LSTM_MinMaxModel

def regression_modal( NowDateTime , AllOutPut , Regression_X_train , Regression_y_train ):
    RegressionModel , LSTM_MinMaxModel = create_modal( AllOutPut , Regression_X_train , Regression_y_train)
    joblib.dump(RegressionModel, './model/WheatherRegression_'+NowDateTime)
    print('intercept: ',RegressionModel.intercept_)

    print('coef : ', RegressionModel.coef_)

    print('R squared: ',RegressionModel.score(LSTM_MinMaxModel.transform(Regression_X_train), Regression_y_train))


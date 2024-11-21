from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def normal( AllOutPut , LookBackNum ):
    LSTM_MinMaxModel = MinMaxScaler().fit(AllOutPut)
    AllOutPut_MinMax = LSTM_MinMaxModel.transform(AllOutPut)
    
    X_train = []
    y_train = []
    
    for i in range(LookBackNum,len(AllOutPut_MinMax)):
        X_train.append(AllOutPut_MinMax[i-LookBackNum:i, :])
        y_train.append(AllOutPut_MinMax[i, :])


    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 10))
    # X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 5))
    return X_train , y_train , LSTM_MinMaxModel

def reshape(X_train):
    print("original size:", X_train.shape)  # ?????��??�?形�??
    print("elements:", X_train.size)  # ?????��??�???��??
    return np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 2))

    
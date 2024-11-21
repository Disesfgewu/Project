# import tensorflow as tf
# tf.get_logger().setLevel('ERROR')

# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Dense, Dropout, Input, MultiHeadAttention, LayerNormalization, Add
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.models import load_model
# from tensorflow.keras.layers import GlobalAveragePooling1D
# from datetime import timedelta

# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.preprocessing import MinMaxScaler
# from datetime import datetime
# import joblib

# import numpy as np
# import pandas as pd
# import os

# def transformer_model(input_shape):
#     inputs = Input(shape=input_shape)
    
#     attention_output = MultiHeadAttention(num_heads=8, key_dim=input_shape[-1])(inputs, inputs)
#     attention_output = Add()([inputs, attention_output])
#     attention_output = LayerNormalization(epsilon=1e-6)(attention_output)

#     ffn_output = Dense(128, activation="relu")(attention_output)
#     ffn_output = Dense(input_shape[-1])(ffn_output)
#     ffn_output = Add()([attention_output, ffn_output])
#     ffn_output = LayerNormalization(epsilon=1e-6)(ffn_output)

#     ffn_output = GlobalAveragePooling1D()(ffn_output)
    
#     outputs = Dense(5)(ffn_output)
    
#     model = Model(inputs, outputs)
#     model.compile(optimizer=Adam(learning_rate=0.001), loss="mean_squared_error")
#     return model

# def train( X_train, y_train, epochs=100, batch_size=64 ):
#     # ¶}©l°V½m
#     regressor = transformer_model((X_train.shape[1], X_train.shape[2]))
    
#     regressor.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

#     # «O¦s¼Ò«¬
#     NowDateTime = datetime.now().strftime("%Y-%m-%dT%H_%M_%SZ")
#     regressor.save('WeatherTransformer.keras')
#     print('Model Saved')

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, MultiHeadAttention, LayerNormalization, Add, LSTM, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import GlobalAveragePooling1D
from datetime import datetime
import joblib

import numpy as np
import pandas as pd
import os

def combined_model(input_shape):
    """
    定義結合 Transformer、Attention、LSTM 和 CNN 的模型
    :param input_shape: 輸入數據的形狀 (時間步長, 特徵數)
    :return: 編譯完成的 Keras 模型
    """
    inputs = Input(shape=input_shape)

    # CNN 部分：局部特徵提取
    cnn_output = Conv1D(filters=128, kernel_size=5, activation='relu')(inputs)
    cnn_output = MaxPooling1D(pool_size=2)(cnn_output)

    # LSTM 部分：捕捉時間序列的依賴性
    lstm_output = LSTM(units=128, return_sequences=True)(cnn_output)
    lstm_output = Dropout(0.3)(lstm_output)

    # Transformer 部分：捕捉全局依賴性
    attention_output = MultiHeadAttention(num_heads=8, key_dim=lstm_output.shape[-1])(lstm_output, lstm_output)
    attention_output = Add()([lstm_output, attention_output])
    attention_output = LayerNormalization(epsilon=1e-6)(attention_output)

    # Feed-Forward Network (FFN)
    ffn_output = Dense(128, activation="relu")(attention_output)
    ffn_output = Dense(128)(ffn_output)  # 調整維度為 128，使其與 attention_output 一致
    ffn_output = Add()([attention_output, ffn_output])
    ffn_output = LayerNormalization(epsilon=1e-6)(ffn_output)

    # 全局池化 + 全連接層
    ffn_output = GlobalAveragePooling1D()(ffn_output)
    outputs = Dense(10)(ffn_output)  # 輸出 5 個特徵

    # 模型定義與編譯
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mean_squared_error")
    return model


def train(X_train, y_train, epochs=100, batch_size=64):
    """
    訓練結合模型
    :param X_train: 訓練數據，形狀為 (樣本數, 時間步長, 特徵數)
    :param y_train: 標籤數據，形狀為 (樣本數, 輸出特徵數)
    :param epochs: 訓練的迭代次數
    :param batch_size: 每批次樣本數
    """
    # 初始化模型
    regressor = combined_model((X_train.shape[1], X_train.shape[2]))
    
    # 開始訓練
    regressor.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    # 保存模型
    NowDateTime = datetime.now().strftime("%Y-%m")
    model_path = f'./model/CombinedTransformer_{NowDateTime}.keras'
    regressor.save(model_path)
    print(f'Model Saved: {model_path}')

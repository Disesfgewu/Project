import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, MultiHeadAttention, LayerNormalization, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import GlobalAveragePooling1D
from datetime import timedelta

from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import joblib

import numpy as np
import pandas as pd
import os

def transformer_model(input_shape):
    inputs = Input(shape=input_shape)
    
    attention_output = MultiHeadAttention(num_heads=8, key_dim=input_shape[-1])(inputs, inputs)
    attention_output = Add()([inputs, attention_output])
    attention_output = LayerNormalization(epsilon=1e-6)(attention_output)

    ffn_output = Dense(128, activation="relu")(attention_output)
    ffn_output = Dense(input_shape[-1])(ffn_output)
    ffn_output = Add()([attention_output, ffn_output])
    ffn_output = LayerNormalization(epsilon=1e-6)(ffn_output)

    ffn_output = GlobalAveragePooling1D()(ffn_output)
    
    outputs = Dense(5)(ffn_output)
    
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mean_squared_error")
    return model

def train( X_train, y_train, epochs=100, batch_size=64 ):
    # ¶}©l°V½m
    regressor = transformer_model((X_train.shape[1], X_train.shape[2]))
    
    regressor.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    # «O¦s¼Ò«¬
    NowDateTime = datetime.now().strftime("%Y-%m-%dT%H_%M_%SZ")
    regressor.save('WeatherTransformer.keras')
    print('Model Saved')

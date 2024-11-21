from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

def simple_rnn_model(input_shape):
    model = Sequential()
    model.add(SimpleRNN(units=64, return_sequences=True, input_shape=input_shape))
    model.add(SimpleRNN(units=32))
    model.add(Dense(units=1))  # 根據輸出調整單位數
    model.compile(optimizer='adam', loss='mse')
    return model

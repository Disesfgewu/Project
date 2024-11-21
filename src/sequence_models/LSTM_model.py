from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def deep_lstm_model(input_shape):
    model = Sequential()
    
    # 第一層 LSTM
    model.add(LSTM(units=128, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))  # 加入 Dropout 層來避免過擬合

    # 第二層 LSTM
    model.add(LSTM(units=64, return_sequences=True))
    model.add(Dropout(0.2))

    # 第三層 LSTM
    model.add(LSTM(units=32))
    model.add(Dropout(0.2))

    # 最後的回歸層
    model.add(Dense(units=1))  # 假設是回歸任務，輸出層單位數設為 1

    # 編譯模型
    model.compile(optimizer='adam', loss='mse')  # 使用均方誤差（MSE）作為損失函數

    return model


import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, GRU
from tensorflow.keras.optimizers import Adam
from datetime import datetime
import numpy as np
import os


# 定义改进的 GRU-CNN 模型
def improved_gru_cnn_model(input_shape):
    print("Initializing GRU-CNN Model")
    model = Sequential()
    model.add(Conv1D(filters=128, kernel_size=5, activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(GRU(units=256, return_sequences=True, activation='tanh'))
    model.add(Dropout(0.3))
    model.add(GRU(units=128, activation='tanh'))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(units=1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model


# 定义训练函数
def train_gru_cnn(X_train, y_train, epochs=100, batch_size=64):
    regressor = improved_gru_cnn_model((X_train.shape[1], X_train.shape[2]))
    history = regressor.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        validation_split=0.2,
        shuffle=True
    )

    now_datetime = datetime.now().strftime("%Y-%m")
    model_path = os.path.abspath(f'GRU_CNN_Model_{now_datetime}.h5')
    regressor.save(model_path, save_format='h5')
    print(f'Model Saved: {model_path}')
    return regressor, history



# # 调用方法示例
# if __name__ == "__main__":
#     # 假设 X_train 和 y_train 已经预处理完成
#     X_train = np.random.rand(1000, 30, 10)  # 1000 个样本，30 个时间步，10 个特征
#     y_train = np.random.rand(1000)          # 1000 个样本的目标值

#     # 训练模型
#     model, training_history = train_gru_cnn(X_train, y_train, epochs=50, batch_size=32)

#     # 打印训练曲线
#     import matplotlib.pyplot as plt
#     plt.plot(training_history.history['loss'], label='Training Loss')
#     plt.plot(training_history.history['val_loss'], label='Validation Loss')
#     plt.legend()
#     plt.title('Training and Validation Loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.show()

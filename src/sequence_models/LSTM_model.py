from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import numpy as np

def create_lstm_model(input_shape):
    """
    创建 LSTM 模型
    :param input_shape: 输入数据的形状 (时间步长, 特征数)
    :return: 创建的 LSTM 模型
    """
    model = Sequential()
    model.add(LSTM(units=128, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=64))
    model.add(Dropout(0.2))  # 添加 Dropout 减少过拟合
    return model


def output_layer_setting(model, output_units):
    """
    添加输出层并编译模型
    :param model: 已创建的 LSTM 模型
    :param output_units: 输出层的单元数
    :return: 编译后的模型
    """
    model.add(Dense(units=output_units))  # 输出层
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')  # 编译模型
    return model 


def train(X_train, y_train, NowDateTime, epochs, batch_size):
    """
    训练 LSTM 模型并保存
    :param X_train: 训练数据输入 (样本数, 时间步长, 特征数)
    :param y_train: 训练数据目标 (样本数, 输出维度)
    :param NowDateTime: 当前时间，用于模型命名
    :param epochs: 训练轮数
    :param batch_size: 批量大小
    """
    # 检查输入数据的形状是否匹配
    input_shape = (X_train.shape[1], X_train.shape[2])
    output_units = y_train.shape[1]  # 根据 y_train 自动设置输出维度

    # 创建模型
    model = create_lstm_model(input_shape)
    model = output_layer_setting(model, output_units)

    # 训练模型
    model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,  # 使用 20% 数据作为验证集
        verbose=1,
        shuffle=True
    )

    # 保存模型
    model_path = f'./model/WeatherLSTM_{NowDateTime}.h5'
    model.save(model_path)
    print(f'Model Saved: {model_path}')

    return model

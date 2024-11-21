from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dense, MultiHeadAttention, LayerNormalization, Add, GlobalAveragePooling1D

def gru_transformer_model(input_shape):
    # GRU部分
    inputs = Input(shape=input_shape)
    gru_output = GRU(units=64, return_sequences=True)(inputs)
    gru_output = GRU(units=32, return_sequences=True)(gru_output)

    # Transformer部分
    attention_output = MultiHeadAttention(num_heads=8, key_dim=gru_output.shape[-1])(gru_output, gru_output)
    attention_output = Add()([gru_output, attention_output])
    attention_output = LayerNormalization(epsilon=1e-6)(attention_output)

    ffn_output = Dense(128, activation="relu")(attention_output)
    ffn_output = Dense(gru_output.shape[-1])(ffn_output)
    ffn_output = Add()([attention_output, ffn_output])
    ffn_output = LayerNormalization(epsilon=1e-6)(ffn_output)

    # Pooling + 輸出
    pooled_output = GlobalAveragePooling1D()(ffn_output)
    outputs = Dense(5)(pooled_output)  # 修改輸出維度為 5

    # 編譯模型
    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


from datetime import datetime

def train(X_train, y_train, epochs=100, batch_size=64):
    """
    訓練 GRU + Transformer 模型
    :param X_train: 訓練集特徵 (numpy array)
    :param y_train: 訓練集標籤 (numpy array)
    :param epochs: 訓練的迭代次數
    :param batch_size: 批量大小
    """
    # 創建模型
    regressor = gru_transformer_model((X_train.shape[1], X_train.shape[2]))
    
    # 訓練模型
    regressor.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    # 保存模型
    NowDateTime = datetime.now().strftime("%Y-%m-%dT%H_%M_%SZ")
    regressor.save('WeatherTransformer.keras')
    # regressor.save(model_name)
    print(f'Model Saved')

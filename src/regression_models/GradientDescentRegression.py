import numpy as np
from joblib import dump
import os

def gradient_descent_regression_modal(timestamp, AllOutPut, X_train, y_train, learning_rate=0.01, max_iters=1000, tolerance=1e-6, batch_size=32):
    
    # 確保 X_train 是 2D 陣列
    if len(X_train.shape) != 2:
        raise ValueError("X_train 應該是形狀為 (num_samples, num_features) 的 2D 陣列。")
    
    # 初始化權重和偏置
    num_features = X_train.shape[1]
    weights = np.zeros(num_features)
    bias = 0
    
    # 計算批次數量
    num_samples = X_train.shape[0]
    num_batches = int(np.ceil(num_samples / batch_size))
    
    # 訓練迴圈
    for i in range(max_iters):
        # 打亂資料順序
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]
        
        for batch_idx in range(num_batches):
            # 取出迷你批次
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            X_batch = X_train_shuffled[start_idx:end_idx]
            y_batch = y_train_shuffled[start_idx:end_idx]
            
            # 預測
            y_pred = np.dot(X_batch, weights) + bias
            
            # 計算梯度
            error = y_pred - y_batch
            gradient_weights = (2 / X_batch.shape[0]) * np.dot(X_batch.T, error)  # 計算每個特徵的梯度
            gradient_bias = (2 / X_batch.shape[0]) * np.sum(error)  # 計算偏置的梯度
            
            # 更新權重和偏置（這裡進行梯度下降）
            weights -= learning_rate * gradient_weights
            bias -= learning_rate * gradient_bias
        
        # 收斂檢查
        if np.linalg.norm(gradient_weights) < tolerance and abs(gradient_bias) < tolerance:
            print(f"梯度下降在第 {i+1} 次迭代時收斂")
            break

    # 儲存訓練好的模型
    model_filename = f'./models/GradientDescentRegression_{timestamp}.joblib'
    os.makedirs(os.path.dirname(model_filename), exist_ok=True)
    dump((weights, bias), model_filename)
    print(f"梯度下降回歸模型已儲存至 {model_filename}")

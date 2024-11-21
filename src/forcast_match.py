import pandas as pd

def calculate(sequential_type = "", regression_type = "", batch_size = 0 , epochs = 0 ):
    # 讀取兩個 CSV 文件，忽略第一行
    df1 = pd.read_csv('./result/upload.csv', skiprows=1)
    df2 = pd.read_csv('./result/LSTM256_200_output.csv', skiprows=1)

    # 從第二列開始選取
    df1_selected = df1.iloc[1:9601, 1]
    df2_selected = df2.iloc[1:9601, 1]

    # 檢查選取後的 DataFrame 形狀是否一致
    if df1_selected.shape != df2_selected.shape:
        print("兩個文件的形狀不一致，無法進行比對。")
    else:
        # 計算兩個 DataFrame 從第二列開始的數值差距
        difference = df1_selected - df2_selected

        # 計算總合計的差距（所有數值的絕對值總和）
        total_difference = difference.abs().sum().sum()

        # 打開並追加結果到 txt 文件中
        with open('./result/output.txt', 'a') as f:  # 使用 'a' 模式進行追加
            # 寫入參數信息
            f.write(f"Regression Type: {regression_type}\n")
            f.write(f"Batch Size: {batch_size}\n")
            f.write(f"Epochs: {epochs}\n\n")
            # 寫入總合計差距
            f.write(f"總合計差距（從第二列開始）：{total_difference}\n\n")

        print("總合計差距 : ",total_difference)
        print()

        return total_difference

# 示例使用
# calculate("Linear Regression", 32, 100)

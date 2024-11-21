import os
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

try:
    # Step 1: 定義要讀取的資料夾路徑
    folder_path = './data/ExampleTrainData(AVG)'  # 替換為你的資料夾路徑

    # Step 2: 讀取資料夾內所有 CSV 檔案
    all_data = pd.DataFrame()  # 用於存儲所有資料的 DataFrame

    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            data = pd.read_csv(file_path)
            all_data = pd.concat([all_data, data], ignore_index=True)  # 合併所有資料

    # Step 3: 檢查合併後的資料內容
    print("合併後的資料：")
    print(all_data.head())
    print(all_data.info())

    # Step 4: 選擇數值欄位並處理缺失值，並排除 Power 和 Serial 欄位
    numeric_data = all_data.select_dtypes(include=['float64', 'int64']).drop(columns=['Power(mW)', 'Serial'], errors='ignore').dropna()

    # Step 5: 標準化資料
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_data)

    # Step 6: 執行 PCA，降維到 2 維
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(scaled_data)

    # Step 7: 繪製 PCA 結果的散佈圖
    sns.scatterplot(x=data_pca[:, 0], y=data_pca[:, 1])
    plt.title('PCA of Combined Input Data')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    # 儲存圖表為 PNG 檔案
    output_path = 'pca_result.png'
    plt.savefig(output_path)
    print(f"圖表已儲存為 {output_path}. 請於 Windows 中打開查看。")

    # Step 8: 顯示主成分的解釋
    explained_variance = pca.explained_variance_ratio_
    print("主成分的解釋變異數比例：")
    for i, var in enumerate(explained_variance):
        print(f'主成分 {i + 1}: {var:.4f}')

    # Step 9: 顯示主成分的成分矩陣
    components = pd.DataFrame(pca.components_, columns=numeric_data.columns)
    print("\n主成分的成分矩陣：")
    print(components)

except Exception as e:
    print(f"發生錯誤: {e}")

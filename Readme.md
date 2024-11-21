### 環境

作業系統 : WSL2 Ubuntu 20.04.06 LTS

Python 3.12

### 安裝套件

```bash
    ./install.sh
```

### 執行程式

```bash
    ./run.sh
```


### 指定sequence model、regression model、batch size、epoch


1. 在 ./src/main.py : 60-63 指定需要跑模型(所有模型集合可參考./src/main.py : 54-57)
2. 所有尚未跑的笛卡兒積請見progress_status.csv中第五欄值為'0'的部分



### 結果呈現
- output.csv 會存在result資料夾中
- (若有需求) output.csv 丟給天氣小幫手前需要手動把 upload.csv 和 output.csv 的第一行（中文）刪除
- 執行./src/status_control.py 中的 status_print() 可得知目前預測進度
- progress_status.npy 用於儲存狀態之numpy array
- progress_status.csv 用於儲存狀態之numpy csv

### 比賽模式
1. 先跑一次想要的組合
2. 確認誤差是否與前次相差不是很多
3. 跑competition_forcast
4. 繳交記得刪第一行



### 目前只能跑
預測 : LSTM 、 Transformer

回歸： Catboost , Elasticnet , Huber , Lasso , Lightbm , Ridge , Xgboost , knn , extratree 
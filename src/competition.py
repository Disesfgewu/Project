import os
import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib

def comp_forcast(AllOutPut , lstm , regression_model , k = 0):
    regressor = load_model( lstm )
    Regression = joblib.load( regression_model )
    regressor.compile(optimizer='adam', loss='mean_squared_error')
    LSTM_MinMaxModel = MinMaxScaler().fit(AllOutPut)
    LookBackNum = 12
    ForecastNum = 48

    data_name = './data/ExampleTestData/upload(noanswer).csv'
    source_data = pd.read_csv(data_name, encoding='utf-8')
    target = ['序號']
    ex_question = source_data[target].values
    inputs = []
    predict_output = []
    predict_power = []
    count = 0

    while count < len(ex_question):
        print('count : ', count)
        LocationCode = int(ex_question[count])
        strLocationCode = str(LocationCode)[-2:]
        if LocationCode < 10:
            strLocationCode = '0' + str(LocationCode)

        DataName = './data/ExampleTrainData(IncompleteAVG)/IncompleteAvgDATA_' + strLocationCode + '.csv'
        SourceData = pd.read_csv(DataName, encoding='utf-8')
        ReferTitle = SourceData[['Serial']].values
        ReferData = SourceData[['WindSpeed(m/s)', 'Pressure(hpa)', 'Temperature(°C)', 'Humidity(%)', 'Sunlight(Lux)']].values
        
        inputs = []

        for DaysCount in range(len(ReferTitle)):
            if str(int(ReferTitle[DaysCount]))[:8] == str(int(ex_question[count]))[:8]:
                TempData = ReferData[DaysCount].reshape(1, -1)
                TempData = LSTM_MinMaxModel.transform(TempData)
                inputs.append(TempData)

        for i in range(ForecastNum):
            if i > 0:
                inputs.append(predict_output[i - 1].reshape(1, 5))

            X_test = []
            X_test.append(inputs[0 + i:LookBackNum + i])
            
            NewTest = np.array(X_test)
            NewTest = np.reshape(NewTest, (NewTest.shape[0], NewTest.shape[1], 5 ))
            predicted = regressor.predict(NewTest)
            predict_output.append(predicted)
            predict_power.append(np.round(Regression.predict(predicted), 2).flatten())
        
        count += 48
        

    df = pd.DataFrame(predict_power, columns=['答案'])
    df.insert(0, '序號', ex_question )
    df.to_csv(f'./result/{k}_output.csv', index=False)
    print('Output CSV File Saved')

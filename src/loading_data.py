import pandas as pd 
import os

def getting_title( sourceCode ):
    f = open( sourceCode + "/Title.txt" , "r" , encoding = 'utf-8' )
    get = f.read()
    f.close()
    return get

def adding_title( sourceFile ):
    file_names = os.listdir( sourceFile )
    for file_name in file_names:
        if file_name.endswith('.csv'):
            with open( sourceFile + "/" + file_name , 'r' ) as f:
                get = f.read()
            with open( sourceFile + "/" + file_name , 'w' , encoding = 'utf-8') as fileWrite:
                fileWrite.write( getting_title( sourceFile ) + '\n' + get)
                print( getting_title( sourceFile ) )

def loading_data( sourceFile = "./data/ExampleTrainData(AVG)" , flag = True) :
    if flag == False :
        DataName = sourceFile
        SourceData = pd.read_csv(DataName, encoding='utf-8')
        return SourceData
    else:
        file_names = os.listdir( sourceFile )
        all = []
        for file_name in file_names:
            if file_name.endswith('modified3.csv'):
                # print( file_name )
                file_path = os.path.join(sourceFile, file_name)
                df = pd.read_csv(file_path, encoding='utf-8')
                all.append(df)
        combined_data = pd.concat(all, ignore_index=True)
        return combined_data
def regression_data(sourceData):
    Regression_X_train = sourceData[['WindSpeed(m/s)', 'Pressure(hpa)', 'Temperature(°C)', 'Humidity(%)', 'Sunlight(Lux)' , 'Hour' , 'Season_weight' , 'Sunlight_time(h)' , 'UV' , 'Cloud']].values
    # Regression_X_train = sourceData[['WindSpeed(m/s)', 'Pressure(hpa)', 'Temperature(°C)', 'Humidity(%)', 'Sunlight(Lux)' ]].values
    
    Regression_y_train = sourceData[['Power(mW)']].values 
    return Regression_X_train, Regression_y_train

def LSTM_data(sourceData):
    return sourceData[['WindSpeed(m/s)', 'Pressure(hpa)', 'Temperature(°C)', 'Humidity(%)', 'Sunlight(Lux)' , 'Hour' , 'Season_weight' , 'Sunlight_time(h)' , 'UV' , 'Cloud'  ]].values
    # return sourceData[['WindSpeed(m/s)', 'Pressure(hpa)', 'Temperature(°C)', 'Humidity(%)', 'Sunlight(Lux)' ]].values


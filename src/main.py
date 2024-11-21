from datetime import datetime
import time
import sys


from loading_data import *
from normalize import *
from forcast import *
from forcast_match import *
from status_control import *
from competition import *

sys.path.append('/sequence_models/')
from sequence_models.tranformation_model import *
from sequence_models.LSTM_model import *
# from sequence_models.gru_model import *
# from sequence_models.simple_rnn_model import *
# from sequence_models.bidirectional_LSTM import *
# from sequence_models.gru_tran import *

sys.path.append('/regression_models/')
from regression import *
from regression_models.VotingRegressor import *
# from regression_models.KnnRegression import *
# from regression_models.ExtraTreesRegressor import *
# from regression_models.RandomForestRegression import *
# from regression_models.GradientBoostingRegression import *
# from regression_models.SupportVectorRegression import *
# from regression_models.GradientDescentRegression import *
from regression_models.XgboostRegression import *
from regression_models.CatboostRegression import *
from regression_models.LightgbmRegression import *
from regression_models.ElasticnetRegression import *
from regression_models.HuberRegression import *
from regression_models.LassoRegression import *
from regression_models.RidgeRegression import *

def main():
    start_time = time.time()
    #adding_title( "./data/ExampleTrainData(AVG)" )
    #adding_title( "./data/ExampleTrainData(IncompleteAVG)" )
    SourceData = loading_data( "./data/ExampleTrainData(AVG)" , True)
    AllOutPut = LSTM_data( SourceData ) 
    Regression_X_train , Regression_y_train = regression_data( SourceData )
    
    X_train , y_train , LSTM_MinMaxModel = normal( AllOutPut , 12 )
    # X_train = reshape( x_train )

    
    
    NowDateTime = datetime.now().strftime("%Y-%m")

    # orgiginal
    # runnung_type = ["try 1st", "try again", "competition"]
    # seq_type = ["Transformer", "LSTM", "GRU", "Simple RNN", "Bidirectional LSTM"]
    # reg_type = ["ExtraTreesRegressor", "KnnRegression", "VotingRegressor", "Linear", "RandomForestRegressor", "GradientBoostingRegressor", "SupportVectorRegressor", "GradientDescentRegression", "XGBoost", "CatBoost", "LightGBM", "ElasticNet", "Huber", "Lasso", "Ridge"]
    # batch_size_option = [256, 128, 64]
    # epoch_option = [50, 100, 150, 200, 250, 300]

    # hopes
    # running_type = "try again"
    running_type = "try again"
    # seq_type = ["Transformer", "GRU", "Bidirectional LSTM", "LSTM", "Simple RNN"]
    seq_type = [  "Transformer" ]
    # reg_type = ["Lasso", "ExtraTreesRegressor", "KnnRegression", "VotingRegressor", "Linear", "RandomForestRegressor", "GradientBoostingRegressor", "SupportVectorRegressor", "XGBoost", "CatBoost", "LightGBM", "ElasticNet", "Huber", "Ridge"]
    reg_type =  [ "VotingRegressor" ]
    batch_size_option = [128, 256]
    epoch_option = [100, 150, 200]
    k = 0 
    if running_type != "competition":
        # print("111")
        for sequential_type in seq_type:

            if sequential_type == "Transformer":
                regressor = transformer_model((X_train.shape[1], X_train.shape[2]))

            elif sequential_type == "LSTM":
                pass
                # regressor = improved_gru_cnn_model((X_train.shape[1], X_train.shape[2]))
            
            elif sequential_type == "GRU":
                regressor = gru_model((X_train.shape[1], X_train.shape[2]))
            
            elif sequential_type == "Simple RNN":
                regressor = simple_rnn_model((X_train.shape[1], X_train.shape[2]))

            elif sequential_type == "Bidirectional LSTM":
                regressor = bidirectional_lstm_model((X_train.shape[1], X_train.shape[2]))
            elif sequential_type == "gru_tran" :
                regressor = gru_transformer_model( (X_train.shape[1], X_train.shape[2]) )
            # print(now_seq)
            for regression_type in reg_type:
                for batch_size in batch_size_option:
                    for epochs in epoch_option:
                        
                        # status = check_status(sequential_type, regression_type, batch_size, epochs)
                        # print(status)

                        # if status == 0.0 or running_type == "try again":
                            print("---now progressing---")
                            print("Running type: ", running_type)
                            print("Sequencial model type: ", sequential_type)
                            print("Regression type: ", regression_type)
                            print("Batch size: ", batch_size)
                            print("Epochs: ", epochs)
                            train( X_train, y_train, NowDateTime , epochs, batch_size)
                            Regression_y_train = Regression_y_train.ravel()
                            if regression_type == "ExtraTreesRegressor":
                                ExtraTree_regression_modal( NowDateTime , AllOutPut , Regression_X_train , Regression_y_train )

                            elif regression_type == "KnnRegression":
                                knn_regression_modal( NowDateTime , LSTM_MinMaxModel, AllOutPut , Regression_X_train , Regression_y_train )
                            
                            elif regression_type == "VotingRegressor":
                                voting_regression_modal( NowDateTime , AllOutPut , Regression_X_train , Regression_y_train )

                            elif regression_type == "Linear":
                                regression_modal( NowDateTime , AllOutPut , Regression_X_train , Regression_y_train )
                            
                            elif regression_type == "RandomForestRegressor":
                                random_forest_regression_modal(NowDateTime, AllOutPut, Regression_X_train, Regression_y_train)

                            elif regression_type == "GradientBoostingRegressor":
                                gradient_boosting_regression_modal(NowDateTime, AllOutPut, Regression_X_train, Regression_y_train)

                            elif regression_type == "SupportVectorRegressor":
                                support_vector_regression_modal(NowDateTime, AllOutPut, Regression_X_train, Regression_y_train)
                            
                            

                            #new
                            elif regression_type == "XGBoost":
                                xgboost_regression_modal(NowDateTime, AllOutPut, Regression_X_train, Regression_y_train)

                            elif regression_type == "CatBoost":
                                catboost_regression_modal(NowDateTime, AllOutPut, Regression_X_train, Regression_y_train)

                            elif regression_type == "LightGBM":
                                lightgbm_regression_modal(NowDateTime, AllOutPut, Regression_X_train, Regression_y_train)

                            elif regression_type == "ElasticNet":
                                elasticnet_regression_modal(NowDateTime, AllOutPut, Regression_X_train, Regression_y_train)

                            elif regression_type == "Huber":
                                huber_regression_modal(NowDateTime, AllOutPut, Regression_X_train, Regression_y_train)

                            elif regression_type == "Lasso":
                                lasso_regression_modal(NowDateTime, AllOutPut, Regression_X_train, Regression_y_train)

                            elif regression_type == "Ridge":
                                ridge_regression_modal(NowDateTime, AllOutPut, Regression_X_train, Regression_y_train)

                            elif regression_type == "GradientDescentRegression":
                                gradient_descent_regression_modal(NowDateTime, AllOutPut, Regression_X_train, Regression_y_train)


                            if running_type != "competition":
                                print("a")
                                forcast( AllOutPut = AllOutPut , lstm = f"./model/WeatherLSTM_{NowDateTime}.h5", regression_model = f'Regression_{NowDateTime}.pkl' , k = sequential_type + str(batch_size) + "_" + str(epochs))
                                # total_difference = calculate(sequential_type, regression_type, batch_size, epochs)
                                # comp_forcast( AllOutPut = AllOutPut , lstm = f'GRU_CNN_Model_{NowDateTime}.h5' , regression_model = f'./model/WeatherRegression_{NowDateTime}' , k = sequential_type + str(batch_size) + "_" + str(epochs))
                                # import gc
                                # del regressor
                                # gc.collect()
                                # k += 1
                            if running_type == "try 1st":    
                                print("c")
                                modify_status(sequential_type, regression_type, batch_size, epochs, total_difference)
                                status_print()
                                sort_result()    

            
    else:
        print("b")
        comp_forcast( AllOutPut = AllOutPut , lstm = 'WeatherTransformer.keras' , regression_model = f'./model/WeatherRegression_{NowDateTime}' )


    

                    




    
    end_time = time.time() 
    execution_time = end_time - start_time
    
    print( f"execution time : {execution_time}") 
main()

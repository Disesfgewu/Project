import argparse
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
from sequence_models.LSTM_model import *
from sequence_models.gru_tran import *

sys.path.append('/regression_models/')
# from regression_models.VotingRegressor import *
from regression_models.LassoRegression import *
def parse_arguments():
    parser = argparse.ArgumentParser(description="运行预测程序，支持通过命令行指定参数。")
    parser.add_argument("--epoch", type=int, default=150, help="训练时的 epoch 数量")
    parser.add_argument("--batch_size", type=int, default=128, help="训练时的 batch size")
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # 从命令行获取 epoch 和 batch_size
    batch_size_option = [args.batch_size]
    epoch_option = [args.epoch]

    start_time = time.time()

    SourceData = loading_data("./data/ExampleTrainData(AVG)", True)
    AllOutPut = LSTM_data(SourceData)
    Regression_X_train, Regression_y_train = regression_data(SourceData)
    
    X_train, y_train, LSTM_MinMaxModel = normal(AllOutPut, 12)

    NowDateTime = datetime.now().strftime("%Y-%m")

    running_type = "try again"
    seq_type = ["GRU_AND_CNN"]
    reg_type = ["VotingRegressor"]

    if running_type != "competition":
        for sequential_type in seq_type:
            if sequential_type == "LSTM":
                regressor = deep_lstm_model((X_train.shape[1], X_train.shape[2]))
            else :
                regressor = improved_gru_cnn_model((X_train.shape[1], X_train.shape[2]))
            for regression_type in reg_type:
                for batch_size in batch_size_option:
                    for epochs in epoch_option:
                        print("---now progressing---")
                        print("Running type: ", running_type)
                        print("Sequencial model type: ", sequential_type)
                        print("Regression type: ", regression_type)
                        print("Batch size: ", batch_size)
                        print("Epochs: ", epochs)
                        # train_gru_cnn(X_train, y_train, epochs, batch_size)
                        Regression_y_train = Regression_y_train.ravel()
                        lasso_regression_modal(NowDateTime, AllOutPut, Regression_X_train, Regression_y_train)

                        
                        comp_forcast(AllOutPut=AllOutPut, lstm=f"GRU_CNN_Model_{NowDateTime}-21.h5",
                                regression_model=f'./model/LassoRegression_2024-11.pkl',
                                k=sequential_type + "_" + str(batch_size) + "_" + str(epochs))
                        import gc
                        del regressor
                        gc.collect()

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"execution time : {execution_time}")

main()

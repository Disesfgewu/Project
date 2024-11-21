from sklearn.ensemble import VotingRegressor, RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import MinMaxScaler
import joblib
import os


def create_modal(AllOutPut, Regression_X_train, Regression_y_train):
    """
    创建并训练 Voting Regressor 模型
    :param AllOutPut: 数据归一化模型的拟合基础数据
    :param Regression_X_train: 训练数据的输入特征
    :param Regression_y_train: 训练数据的目标值
    :return: 训练好的 Voting Regressor 模型和 MinMaxScaler
    """
    # 数据归一化
    LSTM_MinMaxModel = MinMaxScaler().fit(AllOutPut)

    # 定义基础模型
    model1 = RandomForestRegressor(
        n_estimators=200, max_depth=8, min_samples_split=5, min_samples_leaf=2, random_state=42
    )
    model2 = ExtraTreesRegressor(
        n_estimators=200, max_depth=8, max_features='sqrt', random_state=42
    )
    model3 = GradientBoostingRegressor(
        n_estimators=150, learning_rate=0.05, max_depth=10, random_state=42
    )
    model4 = XGBRegressor(
        n_estimators=150, learning_rate=0.25, max_depth=4, reg_alpha=0.1, reg_lambda=1.0, random_state=42,
        objective='reg:squarederror'
    )
    model5 = LGBMRegressor(
        n_estimators=200, learning_rate=0.5, max_depth=8, num_leaves=31, random_state=42
    )
    model6 = Ridge(alpha=1.0)  # L2 正则化
    model7 = Lasso(alpha=0.05)  # L1 正则化

    # 创建 Voting Regressor
    RegressionModel = VotingRegressor(
        estimators=[
            ('rf', model1), ('et', model2), ('gbr', model3), 
            ('xgb', model4), ('lgbm', model5), ('ridge', model6), ('lasso', model7)
        ],
        weights=[2, 2, 1.5, 3, 1, 1, 1]  # 设置模型权重，可调整
    )

    # 训练模型
    RegressionModel.fit(LSTM_MinMaxModel.transform(Regression_X_train), Regression_y_train)
    
    return RegressionModel, LSTM_MinMaxModel


def voting_regression_modal(NowDateTime, AllOutPut, Regression_X_train, Regression_y_train):
    """
    创建、训练和保存 Voting Regressor 模型
    :param NowDateTime: 当前时间，用于模型命名
    :param AllOutPut: 数据归一化模型的拟合基础数据
    :param Regression_X_train: 训练数据的输入特征
    :param Regression_y_train: 训练数据的目标值
    """
    # 创建并训练模型
    RegressionModel, LSTM_MinMaxModel = create_modal(AllOutPut, Regression_X_train, Regression_y_train)

    # 保存模型和归一化器
    os.makedirs('./model', exist_ok=True)
    joblib.dump(RegressionModel, f'./model/WeatherRegression_{NowDateTime}')
    joblib.dump(LSTM_MinMaxModel, './model/LSTM_MinMaxModel.pkl')

    # 打印模型分数
    print('Voting Regressor Model R squared: ',
          RegressionModel.score(LSTM_MinMaxModel.transform(Regression_X_train), Regression_y_train))

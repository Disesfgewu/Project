from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Integer, Continuous
import joblib
import os


def create_modal(AllOutPut, Regression_X_train, Regression_y_train):
    """
    创建并优化 Voting Regressor 模型
    :param AllOutPut: 数据归一化模型的拟合基础数据
    :param Regression_X_train: 训练数据的输入特征
    :param Regression_y_train: 训练数据的目标值
    :return: 优化后的 Voting Regressor 模型和 MinMaxScaler
    """
    # 数据归一化
    LSTM_MinMaxModel = MinMaxScaler().fit(AllOutPut)

    # 定义基础模型和超参数搜索空间
    models = {
        'gbr': GradientBoostingRegressor(random_state=42),
        'etr': ExtraTreesRegressor(random_state=42),
        'knn': KNeighborsRegressor(),
        'xgb': XGBRegressor(random_state=42, objective='reg:squarederror'),
        'cat': CatBoostRegressor(random_state=42, verbose=0)
    }
    param_grid = {
        'gbr': {'n_estimators': Integer(100, 500), 'learning_rate': Continuous(0.01, 0.1), 'max_depth': Integer(3, 8)},
        'etr': {'n_estimators': Integer(100, 300), 'max_depth': Integer(5, 15)},
        'knn': {'n_neighbors': Integer(3, 10)},
        'xgb': {'n_estimators': Integer(100, 300), 'learning_rate': Continuous(0.01, 0.1), 'max_depth': Integer(3, 8)},
        'cat': {'iterations': Integer(100, 300), 'learning_rate': Continuous(0.01, 0.1), 'depth': Integer(4, 10)}
    }

    # 基因算法优化每个模型
    optimized_models = {}
    for name, model in models.items():
        print(f"Optimizing {name}...")
        gas = GASearchCV(estimator=model,
                         param_grid=param_grid[name],
                         scoring='r2',
                         cv=3,
                         n_jobs=-1,
                         verbose=0,
                         population_size=10,
                         generations=5)
        gas.fit(LSTM_MinMaxModel.transform(Regression_X_train), Regression_y_train)
        optimized_models[name] = gas.best_estimator_
        print(f"{name} best parameters: {gas.best_params_}")

    # 创建 Voting Regressor
    RegressionModel = VotingRegressor(
        estimators=[
            ('gbr', optimized_models['gbr']),
            ('etr', optimized_models['etr']),
            ('knn', optimized_models['knn']),
            ('xgb', optimized_models['xgb']),
            ('cat', optimized_models['cat']),
        ],
        weights=[2, 1.5, 1, 2.5, 3]  # 可根据重要性调整权重
    )

    # 训练模型
    RegressionModel.fit(LSTM_MinMaxModel.transform(Regression_X_train), Regression_y_train)
    return RegressionModel, LSTM_MinMaxModel


def voting_regression_modal(NowDateTime, AllOutPut, Regression_X_train, Regression_y_train):
    """
    创建、优化和保存 Voting Regressor 模型
    :param NowDateTime: 当前时间，用于模型命名
    :param AllOutPut: 数据归一化模型的拟合基础数据
    :param Regression_X_train: 训练数据的输入特征 (包含 10 个特征)
    :param Regression_y_train: 训练数据的目标值
    """
    # 检查输入特征是否为 10
    if Regression_X_train.shape[1] != 10:
        raise ValueError("输入特征数量必须为 10，请检查数据的形状！")

    # 创建并训练模型
    RegressionModel, LSTM_MinMaxModel = create_modal(AllOutPut, Regression_X_train, Regression_y_train)

    # 保存模型和归一化器
    os.makedirs('./model', exist_ok=True)
    joblib.dump(RegressionModel, f'./model/Regression_{NowDateTime}.pkl')
    joblib.dump(LSTM_MinMaxModel, './model/LSTM_MinMaxModel.pkl')

    # 打印模型分数
    print('Voting Regressor Model R squared: ',
          RegressionModel.score(LSTM_MinMaxModel.transform(Regression_X_train), Regression_y_train))

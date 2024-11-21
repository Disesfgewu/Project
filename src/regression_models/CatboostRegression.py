import joblib
from sklearn.linear_model import ElasticNet, HuberRegressor, Lasso, Ridge
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

def catboost_regression_modal(NowDateTime, AllOutPut, X_train, y_train):
    model = CatBoostRegressor(iterations=100, learning_rate=0.1, depth=6, verbose=0)
    model.fit(X_train, y_train)
    joblib.dump(model, f'./model/CatBoostRegression_{NowDateTime}.pkl')
    print("CatBoost model saved!")

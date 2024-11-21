import joblib
from sklearn.linear_model import ElasticNet, HuberRegressor, Lasso, Ridge
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

def huber_regression_modal(NowDateTime, AllOutPut, X_train, y_train):
    model = HuberRegressor(epsilon=1.35)
    model.fit(X_train, y_train)
    joblib.dump(model, f'./model/HuberRegression_{NowDateTime}.pkl')
    print("Huber model saved!")
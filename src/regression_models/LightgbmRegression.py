import joblib
from sklearn.linear_model import ElasticNet, HuberRegressor, Lasso, Ridge
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

def lightgbm_regression_modal(NowDateTime, AllOutPut, X_train, y_train):
    model = LGBMRegressor(n_estimators=100, learning_rate=0.1)
    model.fit(X_train, y_train)
    joblib.dump(model, f'./model/LightGBMRegression_{NowDateTime}.pkl')
    print("LightGBM model saved!")

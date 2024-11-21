import joblib
from sklearn.linear_model import ElasticNet, HuberRegressor, Lasso, Ridge
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

def ridge_regression_modal(NowDateTime, AllOutPut, X_train, y_train):
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    joblib.dump(model, f'./model/RidgeRegression_{NowDateTime}.pkl')
    print("Ridge model saved!")

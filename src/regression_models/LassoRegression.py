import joblib
from sklearn.linear_model import ElasticNet, HuberRegressor, Lasso, Ridge
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

def lasso_regression_modal(NowDateTime, AllOutPut, X_train, y_train):
    model = Lasso(alpha=0.1)
    model.fit(X_train, y_train)
    joblib.dump(model, f'./model/LassoRegression_{NowDateTime}.pkl')
    print("Lasso model saved!")

import joblib
from sklearn.linear_model import ElasticNet, HuberRegressor, Lasso, Ridge
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

def elasticnet_regression_modal(NowDateTime, AllOutPut, X_train, y_train):
    model = ElasticNet(alpha=0.1, l1_ratio=0.5)
    model.fit(X_train, y_train)
    joblib.dump(model, f'./model/ElasticNetRegression_{NowDateTime}.pkl')
    print("ElasticNet model saved!")

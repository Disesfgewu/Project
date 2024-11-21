from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from joblib import dump
import os

def random_forest_regression_modal(timestamp, AllOutPut, X_train, y_train):

    print("Training RandomForestRegressor...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    model_filename = f'./models/RandomForestRegression_{timestamp}.joblib'
    os.makedirs(os.path.dirname(model_filename), exist_ok=True)
    dump(model, model_filename)
    print(f"RandomForestRegressor model saved to {model_filename}")
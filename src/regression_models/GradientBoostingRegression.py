from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from joblib import dump
import os


def gradient_boosting_regression_modal(timestamp, AllOutPut, X_train, y_train):

    print("Training GradientBoostingRegressor...")
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    
    model_filename = f'./models/GradientBoostingRegression_{timestamp}.joblib'
    os.makedirs(os.path.dirname(model_filename), exist_ok=True)
    dump(model, model_filename)
    print(f"GradientBoostingRegressor model saved to {model_filename}")
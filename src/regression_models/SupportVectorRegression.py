from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from joblib import dump
import os

def support_vector_regression_modal(timestamp, AllOutPut, X_train, y_train):
    
    print("Training SupportVectorRegressor...")
    model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    model.fit(X_train, y_train)
    
    model_filename = f'./models/SupportVectorRegression_{timestamp}.joblib'
    os.makedirs(os.path.dirname(model_filename), exist_ok=True)
    dump(model, model_filename)
    print(f"SupportVectorRegressor model saved to {model_filename}")
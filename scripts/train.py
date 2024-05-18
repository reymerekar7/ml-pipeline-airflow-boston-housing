# This file was created as part of the pipeline to train a linear regression on the Boston housing dataset and package its model artifact

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

class BostonLinearRegression:
    def __init__(self):
        self.model = LinearRegression()
        self.features = None
        self.target = None

    def fit_regression(self, X, y, feature_names=None, target_name=None):
        self.model.fit(X,y)
        self.features = feature_names
        self.target = target_name
        print("Model fit successfully") 

    def predict(self, X):
        return self.model.predict(X)
    
    def evaluate(self, X, y):
        predictions = self.predict(X)
        mse = mean_squared_error(y, predictions)
        rmse = mse ** 0.5
        r2 = r2_score(y, predictions)

        # log current time of evaluation
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

        eval_dict = {'Timestamp': [timestamp], 'MSE': [mse], 'RMSE': [rmse], 'R^2': [r2]}
        eval_df = pd.DataFrame(eval_dict)
        eval_df.to_csv('/Users/reymerekar/Desktop/ml_pipeline_airflow/artifacts/model_performance.csv')
        print("Evaluation stored in csv")
    
    def save_model(self, filename):
        # Save the model
        joblib.dump(self.model, filename)
        print("model artifact saved")

def main():
    
    # Read the data
    column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    data = pd.read_csv('/Users/reymerekar/Desktop/ml_pipeline_airflow/data/boston_housing.csv', header=None, delimiter=r"\s+", names=column_names)

    # Split data
    feature_names = data.columns[:-1]
    target_name = data.columns[-1]
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Instantiate class, fit, evaluate
    boston_regressor = BostonLinearRegression()
    boston_regressor.fit_regression(X_train, y_train, feature_names, target_name)
    boston_regressor.evaluate(X_test, y_test)

    # Package model
    save_path = '/Users/reymerekar/Desktop/ml_pipeline_airflow/artifacts/model.joblib'
    boston_regressor.save_model(save_path)

if __name__ == "__main__":
    main()
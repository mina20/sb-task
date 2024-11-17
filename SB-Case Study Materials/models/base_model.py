import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from bayes_opt import BayesianOptimization
import numpy as np
import pandas as pd

class ModelTuner:
    def __init__(self, df, target_column):
        self.df = df
        self.target_column = target_column
        self.X = df.drop(columns=[target_column])
        self.y = df[target_column]
        
        self.models = {
            'XGBoost': xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
            'Random Forest': RandomForestRegressor(random_state=42),
            'SVR': SVR()
        }
        
    def train_test_split(self, test_size=0.2, random_state=42):
        return train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)

    def bayesian_optimization(self, model, X_train, y_train, param_bounds, n_iter=10):
        def objective_function(**params):
            if 'n_estimators' in params:
                params['n_estimators'] = int(params['n_estimators'])  
            if 'max_depth' in params:
                params['max_depth'] = int(params['max_depth'])
            model.set_params(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_train)
            mse = mean_squared_error(y_train, y_pred) 
            return -mse  

        optimizer = BayesianOptimization(
            f=objective_function,
            pbounds=param_bounds,
            random_state=42
        )
        optimizer.maximize(init_points=5, n_iter=n_iter)
        return optimizer

    def xgboost_model(self, X_train, y_train):
        xgb_param_bounds = {
            'n_estimators': (50, 1000),  # Integer
            'learning_rate': (0.01, 0.3),  # Float
            'max_depth': (3, 15),  # Integer
            'subsample': (0.5, 1),  # Float
            'colsample_bytree': (0.5, 1)  # Float
        }
        optimizer = self.bayesian_optimization(self.models['XGBoost'], X_train, y_train, xgb_param_bounds, n_iter=20)
        best_params = optimizer.max['params']
        best_params['n_estimators'] = int(best_params['n_estimators'])
        best_params['max_depth'] = int(best_params['max_depth'])
        self.models['XGBoost'].set_params(**best_params)
        self.models['XGBoost'].fit(X_train, y_train)
    
        return self.models['XGBoost']

    def random_forest_model(self, X_train, y_train):
        rf_param_bounds = {
            'n_estimators': (50, 1000),  
            'max_depth': (3, 15), 
            'min_samples_split': (2, 10),  
            'min_samples_leaf': (1, 10),  
            'max_features': (0.1, 1.0)  
        }
        
        optimizer = self.bayesian_optimization(self.models['Random Forest'], X_train, y_train, rf_param_bounds, n_iter=20)
        best_params = optimizer.max['params']
        best_params['n_estimators'] = int(best_params['n_estimators'])
        best_params['max_depth'] = int(best_params['max_depth'])
        best_params['min_samples_split'] = int(best_params['min_samples_split'])
        best_params['min_samples_leaf'] = int(best_params['min_samples_leaf'])
        self.models['Random Forest'].set_params(**best_params)
        self.models['Random Forest'].fit(X_train, y_train)
        
        return self.models['Random Forest']

    def svr_model(self, X_train, y_train):
        svr_param_bounds = {
            'C': (1e-3, 1e3),  
            'epsilon': (0.01, 0.1),  
            'kernel': (0, 1)  
        }
        optimizer = self.bayesian_optimization(self.models['SVR'], X_train, y_train, svr_param_bounds, n_iter=20)
        best_params = optimizer.max['params']
        best_params['kernel'] = 'linear' if best_params['kernel'] < 0.5 else 'rbf'
        self.models['SVR'].set_params(**best_params)
        self.models['SVR'].fit(X_train, y_train)
        
        return self.models['SVR']


    def evaluate_models(self, X_test, y_test, model):
        y_pred = model.predict(X_test)
        results_df = pd.DataFrame({
            'y_actual': y_test,
            'y_pred': y_pred
        })
    
        return results_df


    def fit_and_evaluate(self, test_size=0.2, models_to_fit=None):
        if models_to_fit is None:
            models_to_fit = ['XGBoost', 'Random Forest', 'SVR']
        
        X_train, X_test, y_train, y_test = self.train_test_split(test_size)
        evaluation_results = {}
        if 'XGBoost' in models_to_fit:
            print("Training and optimizing XGBoost...")
            xgb_model = self.xgboost_model(X_train, y_train)
            evaluation_results['XGBoost'] = self.evaluate_models(X_test, y_test, xgb_model)
        if 'Random Forest' in models_to_fit:
            print("Training and optimizing Random Forest...")
            rf_model = self.random_forest_model(X_train, y_train)
            evaluation_results['Random Forest'] = self.evaluate_models(X_test, y_test, rf_model)
        
        if 'SVR' in models_to_fit:
            print("Training and optimizing SVR...")
            svr_model = self.svr_model(X_train, y_train)
            evaluation_results['SVR'] = self.evaluate_models(X_test, y_test, svr_model)
        
        return evaluation_results
            
if __name__ == "__main__":
    
    df = pd.read_csv('./../data/pro1.csv')
    target_column = 'annual_amount'  
    tuner = ModelTuner(df, target_column)
    results = tuner.fit_and_evaluate(test_size=0.2,models_to_fit=['XGBoost'])
    print("Evaluation results:", results)

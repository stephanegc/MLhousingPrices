import joblib
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor 
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
import numpy as np

def load_model(filename='MLhousingPrices/resources/xgb_tuned.pkl'):
    pp = joblib.load(filename)
    return pp

class ModelTrainer:
    def __init__(self, modelType='xgb'):
        modelType = modelType.lower()
        if modelType not in ['xgb', 'rfr', 'knn']:
            raise ValueError("modelType needs to be xgb, rfr or knn !")
        self.modelType = modelType

    def train(self, X_train_pp, y_train):
        if self.modelType == 'xgb':
            param_grid = {
                'n_estimators': [5, 10, 100, 500],
                'max_depth': [2, 5, 10, 15],
                'learning_rate': [0.05, 0.1, 0.15, 0.20],
                'min_child_weight': [1, 2, 3, 4]
            }
            reg = XGBRegressor(random_state = 0)
        elif self.modelType == 'rfr':
            param_grid =  {
                'n_estimators' : [5, 10, 100, 500, 500],
                'max_features'  : np.linspace(0.1, 1, num=4)
            }
            reg = RandomForestRegressor(random_state = 0)
        else:
            param_grid = { 
                'n_neighbors' : [2, 5, 10, 15, 30, 50],
                'weights' : ['uniform','distance']
            }
            reg = KNeighborsRegressor()
            
        model = GridSearchCV(estimator = reg, param_grid = param_grid, n_jobs=-1, scoring='r2')
        model.fit(X_train_pp, y_train)
        self.model = model
        return self
    
    def save(self, filename: str):
        joblib.dump(self, f"model_{self.modelType}_{filename}.pkl")



import joblib

def load_model(filename='MLhousingPrices/resources/xgb_tuned.pkl'):
    pp = joblib.load(filename)
    return pp

class ModelTrainer:
    def __init__(self, modelType: str):
        str.modelType = modelType

    def train(self, X_train, y_train):
        pass


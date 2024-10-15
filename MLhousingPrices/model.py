import joblib

def load_model(filename='MLhousingPrices/resources/rfr_tuned.pkl'):
    pp = joblib.load(filename)
    return pp

class ModelTrainer:
    def __init__(self):
        pass

    def train(self, X_train, y_train):
        pass


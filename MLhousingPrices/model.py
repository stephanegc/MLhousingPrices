import joblib

def load_model(filename='MLhousingPrices/resources/lr_0.pkl'):
    pp = joblib.load(filename)
    return pp
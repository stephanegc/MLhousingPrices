import joblib

def load_preprocessor(filename='MLhousingPrices/resources/preprocessor_0.pkl'):
    pp = joblib.load(filename)
    return pp
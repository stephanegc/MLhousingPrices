import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import re

def load_preprocessor(filename='MLhousingPrices/resources/preprocessor_03.pkl'):
    pp = joblib.load(filename)
    return pp

class Preprocessor():
    def __init__(self, data, subset=True, cluster=False):
        self.dataImport = data
        
        if 'median_house_value' in self.dataImport.columns:
            self.y = self.dataImport.median_house_value

        if subset:
            self.X = self.dataImport[['longitude', 'median_income', 'latitude', 'ocean_proximity']]
        else:
            self.X = self.dataImport.drop('median_house_value', axis=1)
    
        if cluster:
            km = KMeans(n_clusters=8)
            km.fit(self.X[['latitude', 'longitude']])
            self.X['cluster'] = km.fit_predict(self.X[['latitude', 'longitude']])
            self.X['cluster'] = self.X['cluster'].astype('category')

        X_cont = self.X.select_dtypes(include=['int64', 'float64'])
        X_cat = self.X.select_dtypes(include=['object'])

        numeric_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median")), 
                   ("scaler", StandardScaler())]
        )
        categorical_transformer = Pipeline(
            steps=[
                ("encoder", OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]
        )
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, X_cont.columns),
                ("cat", categorical_transformer, X_cat.columns),
            ]
        )
        self.preprocessor.set_output(transform='pandas')

    def split(self):
        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(self.X, self.y, train_size=0.8, test_size=0.2, random_state=0)
        return self

    def preprocess_train(self):
        self.X_train_pp = self.preprocessor.fit_transform(self.X_train)
        self.X_valid_pp = self.preprocessor.transform(self.X_valid)
        self.X_train_pp = self.rename_cols(self.X_train_pp)
        self.X_valid_pp = self.rename_cols(self.X_valid_pp)
        return self

    def preprocess_test(self, X_test):
        self.X_test_pp = self.preprocessor.transform(X_test)
        self.X_test_pp = self.rename_cols(self.X_test_pp)
        return self  
    
    def rename_cols(self, df):
        regex = re.compile(r"\[|\]|<", re.IGNORECASE)
        df.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in df.columns.values]
        return df
    
    def save(self, filename: str):
        joblib.dump(self, f"preprocessor_{filename}.pkl")
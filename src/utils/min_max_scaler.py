from sklearn.preprocessing import MinMaxScaler as _MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class MinMaxScaler(BaseEstimator, TransformerMixin):

    def __init__(self, cols_list=["Open", "High", "Low", "Close"]):
        self.cols_list = cols_list
        self.scaler = ColumnTransformer(
            transformers=[
                ('cat', _MinMaxScaler(), self.cols_list)
            ], remainder='passthrough'
        )

    def fit(self, X, y=None):
        self.scaler.fit(X)
        return self

    def transform(self, X, y=None):
        _X = X.copy()
        _X = self.scaler.transform(_X)
        _X = pd.DataFrame(_X, index=X.index, columns=X.columns)

        return _X
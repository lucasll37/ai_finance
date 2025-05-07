from sklearn.base import BaseEstimator, TransformerMixin

class MakeTimeSeries(BaseEstimator, TransformerMixin):

    def __init__(self, window=10, indicators=None):
        self.window = window
        self.indicators = indicators

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        _X = X.copy()

        features = ['_Open', '_High', '_Low', '_Close', '_Volume']

        if self.indicators is None:
            self.indicators = [col for col in _X.columns if (not col.startswith("_")) and (col != "Trend")]


        for indicator in self.indicators:
            ind_type = _X[indicator].dtype

            for i in range(self.window-2, -1, -1):
                feature = f"{indicator} D-{i+1}"
                features.append(feature)

                _X.loc[_X.index[i+1]: , feature] = _X.loc[:_X.index[-(i+2)], indicator].values
                _X.loc[_X.index[i+1]: , feature] = _X.loc[_X.index[i+1]: , feature].astype(ind_type)

            feature = f"{indicator} D-0"
            features.append(feature)
            _X[feature] = _X.loc[:, indicator].values
            _X.drop([indicator], axis=1, inplace=True)

        _X = _X[features + ["Trend"]]
        _X.dropna(axis=0, inplace=True)

        return _X
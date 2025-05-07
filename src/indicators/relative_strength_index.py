import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

sns.set_theme(style='ticks')


class RelativeStrengthIndex(BaseEstimator, TransformerMixin):

    def __init__(self, window=14, graphic=True, path="../figures/RSI_Graphic.png"):
        self.window = window
        self.graphic = graphic
        self.path = path

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        _X = X.copy()
        delta = _X['_Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.window).mean()
        RS = gain / loss
        RSI = 100 - (100 / (1 + RS))
        _X['RSI'] = RSI

        if self.graphic:
            plt.figure(figsize=(15, 5), dpi=100)
            sns.lineplot(data=_X, x='Date', y='RSI', color='b', label='RSI')
            plt.axhline(70, linestyle='--', color='r')
            plt.axhline(30, linestyle='--', color='r')
            plt.xlabel('Date')
            plt.ylabel('RSI')
            plt.title('Relative Strength Index (RSI)')
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.path)
            plt.close()
            
        return _X

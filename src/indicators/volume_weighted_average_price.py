import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

sns.set_theme(style='ticks')

class VolumeWeightedAveragePrice(BaseEstimator, TransformerMixin):

    def __init__(self, graphic=True, path="../figures/VWAP_Graphic.png"):
        self.graphic = graphic
        self.path = path

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        _X = X.copy()
        _X['VWAP'] = (_X['_Volume'] * _X['_Close']).cumsum() / _X['_Volume'].cumsum()

        if self.graphic:
            fig, ax = plt.subplots(figsize=(15, 5), dpi=100)
            sns.lineplot(data=_X, x='Date', y='_Close', label='Close', color='gray', linestyle='-', linewidth=1, alpha=0.2, ax=ax)
            sns.lineplot(data=_X, x='Date', y='VWAP', label='VWAP', color='blue', linestyle='-', linewidth=2, ax=ax)
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.legend()
            fig.suptitle('Volume Weighted Average Price (VWAP)', fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(self.path)
            plt.close()

        return _X

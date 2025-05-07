import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

sns.set_theme(style='ticks')


class AwesomeOscillator(BaseEstimator, TransformerMixin):

    def __init__(self, short=5, long=34, graphic=True, path="../figures/AO_Graphic.png"):
        self.short = short
        self.long = long
        self.graphic = graphic
        self.path = path

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        _X = X.copy()
        _X['_AO_Median'] = (_X['_High'] + _X['_Low']) / 2
        _X['AO'] = _X['_AO_Median'].rolling(window=self.short).mean() - _X['_AO_Median'].rolling(window=self.long).mean()

        if self.graphic:
            fig, ax = plt.subplots(figsize=(15, 5), dpi=100)

            sns.lineplot(_X, x="Date", y="AO", label="AO", linestyle='-', linewidth=1, color="black", alpha=1, ax=ax, legend=False)
            ax.axhline(0, linestyle='--', color='r')
            ax.set_xlabel('Date')
            ax.set_ylabel('Awesome Oscillator (AO)')
            ax.set_title('Awesome Oscillator (AO)')

            plt.tight_layout()
            plt.savefig(self.path)
            plt.close()

        return _X

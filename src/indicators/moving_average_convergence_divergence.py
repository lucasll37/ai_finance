import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

sns.set_theme(style='ticks')


class MovingAverageConvergenceDivergence(BaseEstimator, TransformerMixin):

    def __init__(self, short=12, long=26, signal=9, graphic=True, path="../figures/MACD_Graphic.png"):
        self.short = short
        self.long = long
        self.signal = signal
        self.graphic = graphic
        self.path = path

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        _X = X.copy()

        short_ema = _X['_Close'].ewm(span=self.short, min_periods=1, adjust=False).mean()
        long_ema = _X['_Close'].ewm(span=self.long, min_periods=1, adjust=False).mean()

        _X['_MACD_Line'] = short_ema - long_ema
        _X['_MACD_Signal_Line'] = _X['_MACD_Line'].ewm(span=self.signal, min_periods=1, adjust=False).mean()
        _X['MACD'] = _X['_MACD_Line'] - _X['_MACD_Signal_Line']

        if self.graphic:
            fig, ax = plt.subplots(figsize=(15, 5), dpi=100)

            sns.lineplot(data=_X, x="Date", y="_MACD_Line", label="MACD Line", linestyle='-', linewidth=1, color="blue", ax=ax)
            sns.lineplot(data=_X, x="Date", y="_MACD_Signal_Line", label="Signal Line", linestyle='-', linewidth=1, color="red", ax=ax)
            sns.lineplot(data=_X, x="Date", y="MACD", color="green", ax=ax)
            # sns.barplot(data=_X, x="Date", y="MACD", color="green", ax=ax)

            ax.axhline(0, linestyle='--', color='gray')
            ax.set_xlabel('Date')
            ax.set_ylabel('MACD')
            ax.set_title('MACD (Moving Average Convergence Divergence)')
            ax.legend()

            plt.tight_layout()
            plt.savefig(self.path)
            plt.close()

        return _X

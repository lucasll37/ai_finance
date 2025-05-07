import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

sns.set_theme(style='ticks')


class AverageDirectionalIndex(BaseEstimator, TransformerMixin):

    def __init__(self, window=14, threshold=20, graphic=True, path="../figures/ADX_Graphic.png"):
        self.window = window
        self.graphic = graphic
        self.path = path
        self.threshold = threshold

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        _X = X.copy()

        # Calculating True Range (TR)
        _X['_ADX_High_Low'] = _X['_High'] - _X['_Low']
        _X['_ADX_High_Close'] = abs(_X['_High'] - _X['_Close'].shift(1))
        _X['_ADX_Low_Close'] = abs(_X['_Low'] - _X['_Close'].shift(1))
        _X['_ADX_True_Range'] = _X[['_ADX_High_Low', '_ADX_High_Close', '_ADX_Low_Close']].max(axis=1)

        # Calculating Directional Movement (DM)
        _X['_ADX_UpMove'] = _X['_High'] - _X['_High'].shift(1)
        _X['_ADX_DownMove'] = _X['_Low'].shift(1) - _X['_Low']
        _X.loc[_X['_ADX_UpMove'] <= _X['_ADX_DownMove'], '_ADX_UpMove'] = 0
        _X.loc[_X['_ADX_DownMove'] <= _X['_ADX_UpMove'], '_ADX_DownMove'] = 0

        # Calculating Smoothed True Range (ATR)
        _X['_ADX_ATR'] = _X['_ADX_True_Range'].rolling(window=self.window).mean()

        # Calculating Positive Directional Movement (+DM) and Negative Directional Movement (-DM)
        _X['_ADX_+DM'] = _X['_ADX_UpMove']
        _X['_ADX_-DM'] = _X['_ADX_DownMove']
        _X.loc[_X['_ADX_+DM'] <= _X['_ADX_-DM'], '_ADX_+DM'] = 0
        _X.loc[_X['_ADX_-DM'] <= _X['_ADX_+DM'], '_ADX_-DM'] = 0

        # Calculating Smoothed Directional Movement (Smoothed +DM and Smoothed -DM)
        _X['_ADX_Smoothed_+DM'] = _X['_ADX_+DM'].rolling(window=self.window).mean()
        _X['_ADX_Smoothed_-DM'] = _X['_ADX_-DM'].rolling(window=self.window).mean()

        # Calculating Directional Index (DI+ and DI-)
        _X['_ADX_DI+'] = (_X['_ADX_Smoothed_+DM'] / _X['_ADX_ATR']) * 100
        _X['_ADX_DI-'] = (_X['_ADX_Smoothed_-DM'] / _X['_ADX_ATR']) * 100

        # Calculating Directional Movement Index (DX)
        _X['_ADX_DX'] = (abs(_X['_ADX_DI+'] - _X['_ADX_DI-']) / (abs(_X['_ADX_DI+']) + abs(_X['_ADX_DI-']))) * 100

        # Calculating Average Directional Index (ADX)
        _X['_ADX'] = _X['_ADX_DX'].rolling(window=self.window).mean()

        _X['_ADX_DI+_minus_DI-'] = _X['_ADX_DI+'] - _X['_ADX_DI-']
        _X['ADX'] = 0

        for index, row in _X.iterrows():
            uptrend = ((row['_ADX'] > self.threshold) & (row['_ADX_DI+_minus_DI-'] > 0))
            downtrend = ((row['_ADX'] > self.threshold) & (row['_ADX_DI+_minus_DI-'] <= 0))

            if uptrend:
                _X.loc[index, 'ADX'] = 1

            elif downtrend:
                _X.loc[index, 'ADX'] = -1

        if self.graphic:
            fig, ax = plt.subplots(figsize=(15, 5), dpi=100)

            ax2 = ax.twinx()

            sns.lineplot(data=_X, x="Date", y="ADX", label="ADX", linestyle='-.', linewidth=1, color="r", alpha=1, ax=ax2, legend=False)
            sns.lineplot(data=_X, x="Date", y="_ADX", label="ADX", linestyle='--', linewidth=2, color="gray", ax=ax, legend=False)
            sns.lineplot(data=_X, x="Date", y='_ADX_DI+_minus_DI-', label="Ddiff", color="blue", ax=ax, legend=False)
            ax.axhline(y=self.threshold, color='gray', linestyle='--', label=f'ADX Threshold')
            ax.axhline(y=0, color='blue', linestyle='--', label=f'Ddiff Threshold')

            ax2.set_ylabel("ADX", color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            
            _y = [-1, 0, 1]
            _labels = {-1: 'Down', 0: 'Neutral', 1: 'Up'}
            ax2.set_yticks(_y)
            ax2.set_yticklabels([_labels[i] for i in _y])

            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

            ax.set_xlabel('Date')
            ax.set_ylabel('ADX')
            ax.set_title('Average Directional Index (ADX)')
            ax.legend()

            plt.tight_layout()
            plt.savefig(self.path)
            plt.close()

        return _X

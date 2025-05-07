import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

sns.set_theme(style='ticks')

class ParabolicSAR(BaseEstimator, TransformerMixin):
    
    def __init__(self, acceleration=0.02, maximum=0.2, graphic=True, path="../figures/Parabolic_SAR_Graphic.png"):
        self.acceleration = acceleration
        self.maximum = maximum
        self.graphic = graphic
        self.path = path

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        _X = X.copy()
        _X['_SAR'] = 0.0
        _X['_EP'] = _X['_High'][0]
        _X['_AF'] = self.acceleration
        _X['pSAR_ind'] = None

        uptrend = True
        _X.loc[_X.index[0], '_SAR'] = _X.loc[_X.index[0], '_Low']

        for i in range(1, len(_X)):
            prev_row = _X.iloc[i-1]
            if uptrend:
                _SAR = prev_row['_SAR'] + prev_row['_AF'] * (prev_row['_EP'] - prev_row['_SAR'])
                if _SAR > _X.loc[_X.index[i], '_Low']:
                    uptrend = False
                    _SAR = prev_row['_EP']
                    _X.loc[_X.index[i], '_AF'] = self.acceleration

                _X.loc[_X.index[i], '_EP'] = prev_row['_EP']

            else:
                _SAR = prev_row['_SAR'] - prev_row['_AF'] * (prev_row['_SAR'] - prev_row['_EP'])
                if _SAR < _X.loc[_X.index[i], '_High']:
                    uptrend = True
                    _SAR = prev_row['_EP']
                    _X.loc[_X.index[i], '_AF'] = self.acceleration

                _X.loc[_X.index[i], '_EP'] = prev_row['_EP']

            _X.loc[_X.index[i], '_SAR'] = _SAR
            _X.loc[_X.index[i], 'pSAR_ind'] = (_X.loc[_X.index[i], '_Low'] - _X.loc[_X.index[i], '_SAR']) / _X.loc[_X.index[i], '_Close']

            if uptrend:
                if _X.loc[_X.index[i], '_High'] > _X.loc[_X.index[i], '_EP']:
                    _X.loc[_X.index[i], '_EP'] = _X.loc[_X.index[i], '_High']
                    _X.loc[_X.index[i], '_AF'] = min(_X.loc[_X.index[i], '_AF'] + self.acceleration, self.maximum)

            else:
                if _X.loc[_X.index[i], '_Low'] < _X.loc[_X.index[i], '_EP']:
                    _X.loc[_X.index[i], '_EP'] = _X.loc[_X.index[i], '_Low']
                    _X.loc[_X.index[i], '_AF'] = min(_X.loc[_X.index[i], '_AF'] + self.acceleration, self.maximum)

        if self.graphic:
            fig, ax0 = plt.subplots(figsize=(15, 5), dpi=100)
            ax1 = ax0.twinx()
            axs = [ax0, ax1]

            sns.lineplot(data=_X, x=_X.index, y='_Close', label='Close', color='blue', linestyle='--', linewidth=2, alpha=0.2, ax=axs[0], legend=False)
            sns.lineplot(data=_X, x=_X.index, y='pSAR_ind', label='pSAR_ind', color='red', linestyle='-', linewidth=1, alpha=1, ax=axs[1], legend=False)
            sns.scatterplot(data=_X, x=_X.index, y='_SAR', label='Parabolic _SAR', color='green', s=50, ax=axs[0], legend=False)
            axs[1].axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
            axs[0].set_xlabel('Date')
            axs[0].set_ylabel('Price')

            axs[1].set_ylabel('pSAR Indicator', color='r')
            axs[1].tick_params(axis='y', labelcolor='r')
            axs[1].set_ylim(-0.2, 0.2)

            lines1, labels1 = axs[0].get_legend_handles_labels()
            lines2, labels2 = axs[1].get_legend_handles_labels()

            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

            fig.suptitle('Parabolic Stop and Reverse (_SAR)', fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(self.path)
            plt.close()

        return _X

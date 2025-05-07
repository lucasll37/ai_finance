import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin

sns.set_theme(style='ticks')

class BollingerBands(BaseEstimator, TransformerMixin):

    def __init__(self, window=20, entry_threshold=0.5, out_tp_threshold=1.5, out_sl_threshold=0, graphic=True, path="../figures/Bollinger_Bands_Graphic.png"):
        self.window = window
        self.graphic = graphic
        self.path = path
        self.entry_threshold = entry_threshold
        self.out_tp_threshold = out_tp_threshold
        self.out_sl_threshold = out_sl_threshold

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        _X = X.copy()

        _X['_BB_Mean'] = _X['_Close'].rolling(window=self.window).mean()
        _X['_BB_Std'] = _X['_Close'].rolling(window=self.window).std()
        _X['_BB_Upper'] = _X['_BB_Mean'] + 2 * _X['_BB_Std']
        _X['_BB_Lower'] = _X['_BB_Mean'] - 2 * _X['_BB_Std']
        _X['_BB_deviation'] = (_X['_Close'] - _X['_BB_Mean']) / _X['_BB_Std']

        _X['BB'] = 0

        temp = 0

        for index, row in _X.iterrows():
            if row['_BB_deviation'] < self.out_sl_threshold:
                temp = 0

            elif row['_BB_deviation'] > self.out_tp_threshold:
                temp = 0

            elif row['_BB_deviation'] > self.entry_threshold:
                temp = 1

            _X.loc[index, 'BB'] = temp


        if self.graphic:
            fig, ax = plt.subplots(figsize=(15, 5), dpi=100)

            ax2 = ax.twinx()

            sns.lineplot(data=_X, x="Date", y="_Close", label="Close", color="black", ax=ax, legend=False)
            sns.lineplot(data=_X, x="Date", y="_BB_Mean", label="BB Mean", color="blue", ax=ax, legend=False)
            sns.lineplot(data=_X, x="Date", y="_BB_Upper", label="BB Upper", color="red", ax=ax, legend=False)
            sns.lineplot(data=_X, x="Date", y="_BB_Lower", label="BB Lower", color="green", ax=ax, legend=False)
            sns.lineplot(data=_X, x="Date", y="_BB_deviation", label="BB", color="red", alpha=0.3, ax=ax2, legend=False)
            sns.lineplot(data=_X, x="Date", y="BB", label="BB", color="blue", alpha=0.3, ax=ax2, legend=False)
            ax2.axhline(y=self.entry_threshold, color='red', linestyle='--', label=f'BBdev Threshold')


            ax2.set_ylabel("deviation", color='r')
            ax2.tick_params(axis='y', labelcolor='r')

            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

            ax.fill_between(_X.index, _X["_BB_Lower"], _X["_BB_Upper"], color='gray', alpha=0.1)
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.set_title('Bollinger Bands')
            ax.legend()

            plt.tight_layout()
            plt.savefig(self.path)
            plt.close()

        return _X

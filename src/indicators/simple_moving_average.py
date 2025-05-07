import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin

sns.set_theme(style='ticks')


class SimpleMovingAverage(BaseEstimator, TransformerMixin):

    def __init__(self, short=3, middle=7, long=21, graphic=True, path="../figures/SMA_Graphic.png"):
        self.short = short
        self.middle = middle
        self.long = long
        self.graphic = graphic
        self.path = path

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        _X = X.copy()
        _X[f'_SMA_{self.short}'] = _X['_Close'].rolling(window=self.short).mean()
        _X[f'_SMA_{self.middle}'] = _X['_Close'].rolling(window=self.middle).mean()
        _X[f'_SMA_{self.long}'] = _X['_Close'].rolling(window=self.long).mean()
        _X['SMA'] = 0

        for index, row in _X.iterrows():
            uptrend = (row[f'_SMA_{self.short}'] - row[f'_SMA_{self.middle}'] > 0) & (row[f'_SMA_{self.middle}'] - row[f'_SMA_{self.long}'] > 0)
            downtrend = (row[f'_SMA_{self.short}'] - row[f'_SMA_{self.middle}'] < 0) & (row[f'_SMA_{self.middle}'] - row[f'_SMA_{self.long}'] < 0)

            if uptrend:
                _X.loc[index, 'SMA'] = 1

            elif downtrend:
                _X.loc[index, 'SMA'] = -1


        if self.graphic:
            fig, ax1 = plt.subplots(figsize=(15, 5), dpi=100)
            ax2 = ax1.twinx()

            axs = [ax1, ax2]

            sns.lineplot(_X, x="Date", y="_Close", label="Close", linestyle='-', linewidth=1, color="gray", alpha=0.2, ax=axs[0], legend=False)
            sns.lineplot(_X, x="Date", y=f"_SMA_{self.long}", label=f"_SMA_{self.long}", linestyle='-', linewidth=2, color="#006400", ax=axs[0], legend=False)
            sns.lineplot(_X, x="Date", y=f"_SMA_{self.middle}", label=f"_SMA_{self.middle}", linestyle='--', linewidth=1, color="#228B22", ax=axs[0], legend=False)
            sns.lineplot(_X, x="Date", y=f"_SMA_{self.short}", label=f"_SMA_{self.short}", linestyle='-', linewidth=0.5, color="blue", ax=axs[0], legend=False)
            axs[0].set_xlabel('Date')
            axs[0].set_ylabel('Price')

            sns.lineplot(_X, x="Date", y="SMA", label="SMA", linestyle='-.', linewidth=1, color="r", alpha=1, ax=axs[1], legend=False)
            axs[1].set_ylabel('SMA', color='r')
            axs[1].tick_params(axis='y', labelcolor='r')

            _y = [-1, 0, 1]
            _labels = {-1: 'Down', 0: 'Neutral', 1: 'Up'}
            axs[1].set_yticks(_y)
            axs[1].set_yticklabels([_labels[i] for i in _y])

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

            fig.suptitle('Simple Moving Average (SMA)', fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.98])

            plt.savefig(self.path)
            plt.close()

        return _X
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin

sns.set_theme(style='ticks')


class Aleatory(BaseEstimator, TransformerMixin):

    def __init__(self, lack_trend=150, graphic=True, path="../figures/Aleatory_Graphic.png"):
        self.lack_trend = lack_trend
        self.graphic = graphic
        self.path = path

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        _X = X.copy()
        position = 0
        current_trend = 0
        limit = len(_X)
        trend = []

        while position < limit:
            lack = np.random.randint(5, self.lack_trend)

            if position + lack < limit:
                trend.extend([current_trend] * lack)
                position += lack
                current_trend = int((not current_trend))

            else:
                lack = limit - position
                trend.extend([current_trend] * lack)
                break

        _X['Aleatory'] = trend
        _X['_Aleatory_entry'] = None
        _X['_Aleatory_out'] = None

        current_trend = _X['Aleatory'][0]
        current = 0

        if current_trend == 1:
            _X['_Aleatory_entry'] = _X['_Close'][0]


        for index, row in _X.iterrows():

            if row['Aleatory'] != current_trend:
                current_trend = int((not current_trend))
                current += 1

                if current_trend == 1:
                    _X.loc[index, '_Aleatory_entry'] = _X.loc[index, '_Close']

                else:
                    _X.loc[index, '_Aleatory_out'] = _X.loc[index, '_Close']

            _X.loc[index, "_Aleatory_Wave"] = current
        
        if self.graphic:
            fig, axs = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={'height_ratios': [3, 1]}, dpi=100)

            sns.lineplot(_X, x="Date", y="_Close", label="_Close", linestyle='-', linewidth=1, color="blue", alpha=1, ax=axs[0], legend=False)
            axs[0].set_xlabel('Date')
            axs[0].set_ylabel('Price')

            sns.lineplot(_X, x="Date", y="Aleatory", label="Aleatory", linestyle='-', linewidth=2, color="black", alpha=1, ax=axs[1], legend=False)
            axs[1].set_ylabel('Uptrand')
            axs[1].set_ylabel('Trend', color='r')
            axs[1].tick_params(axis='y')

            _y = [0, 1]
            _labels = {0: 'Down/Neutral', 1: 'Up'}
            axs[1].set_yticks(_y)
            axs[1].set_yticklabels([_labels[i] for i in _y])

            lines1, labels1 = axs[0].get_legend_handles_labels()
            lines2, labels2 = axs[1].get_legend_handles_labels()
            axs[0].legend(lines1 + lines2, labels1 + labels2, loc='upper left')

            axs[0].scatter(_X.index, _X['_Aleatory_out'], color='green', label='Ponto de saída', marker='v', s=100)
            axs[0].scatter(_X.index, _X['_Aleatory_entry'], color='red', label='Ponto de entrada', marker='^', s=100)

            color = {0: "red", 1: "green"}

            for _, group_data in _X.groupby('_Aleatory_Wave'):
                if group_data['Aleatory'].iloc[0] == 1:
                    axs[0].fill_between(group_data.index, group_data['_Close'].min(), group_data['_Close'].max(), color="green", alpha=0.2)

            fig.suptitle('Aleatory Graphic', fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.98])

            plt.savefig(self.path)
            plt.close()

        return _X
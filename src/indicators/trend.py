import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin

sns.set_theme(style='ticks')


class Trend(BaseEstimator, TransformerMixin):

    def __init__(self, short=7, meddle=21, long=49, graphic=True, path="../figures/Trend_Graphic.png"):
        self.short = short
        self.meddle = meddle
        self.long = long
        self.graphic = graphic
        self.path = path

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        _X = X.copy()
        _X[f'_Trend_SMA_{self.short}'] = _X['_Close'].rolling(window=self.short).mean()
        _X[f'_Trend_SMA_{self.meddle}'] = _X['_Close'].rolling(window=self.meddle).mean()
        _X[f'_Trend_SMA_{self.long}'] = _X['_Close'].rolling(window=self.long).mean()
        _X['Trend'] = None
        _X['_Trend_Group'] = None
        _X['_Max'] = None
        _X['_Min'] = None

        for index, row in _X.iterrows():
            uptrend = (row[f'_Trend_SMA_{self.short}'] > row[f'_Trend_SMA_{self.meddle}']) and \
                (row[f'_Trend_SMA_{self.meddle}'] > row[f'_Trend_SMA_{self.long}']) 
            
            downtrend = (row[f'_Trend_SMA_{self.short}'] < row[f'_Trend_SMA_{self.meddle}']) and \
                (row[f'_Trend_SMA_{self.meddle}'] < row[f'_Trend_SMA_{self.long}']) 

            if uptrend:
                _X.loc[index, 'Trend'] = 1

            elif downtrend:
                _X.loc[index, 'Trend'] = -1

            else:
                _X.loc[index, 'Trend'] = 0

        _X['_Trend_Group'] = (_X['Trend'].diff(1) != 0).astype('int').cumsum()

        for _, group_data in _X.groupby('_Trend_Group'):
            
            min_idx = group_data['_Close'].idxmin()
            max_idx = group_data['_Close'].idxmax()

            if group_data['Trend'].iloc[0] == 1:
                _X.loc[min_idx, '_Min'] = group_data['_Close'].loc[min_idx]
                _X.loc[max_idx, '_Max'] = group_data['_Close'].loc[max_idx]
            
            elif group_data['Trend'].iloc[0] == -1:
                _X.loc[min_idx, '_Min'] = group_data['_Close'].loc[min_idx]
                _X.loc[max_idx, '_Max'] = group_data['_Close'].loc[max_idx]


        for _, group_data in _X.groupby('_Trend_Group'):

            trend = group_data['Trend'].iloc[0]

            if trend == 0:
                continue

            min_idx = group_data['_Close'].idxmin()
            max_idx = group_data['_Close'].idxmax()

            _X.loc[group_data.index, 'Trend'] = 0

            if trend == 1:
                _X.loc[min_idx:max_idx, 'Trend'] = 1

            elif trend == -1:
                _X.loc[max_idx:min_idx, 'Trend'] = -1

            _X['_Trend_Group'] = (_X['Trend'].diff(1) != 0).astype('int').cumsum()

        _X = _X.reindex(columns=[c for c in _X.columns if c != "Trend"] + ["Trend"])
        
        if self.graphic:
            fig, axs = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={'height_ratios': [3, 1]}, dpi=100)

            sns.lineplot(_X, x="Date", y="_Close", label="_Close", linestyle='-', linewidth=1, color="black", alpha=0.5, ax=axs[0], legend=False)
            sns.lineplot(_X, x="Date", y=f"_Trend_SMA_{self.short}", label=f"_Trend_SMA_{self.short}", linestyle='--', linewidth=1, color="green", ax=axs[0], legend=False)
            sns.lineplot(_X, x="Date", y=f"_Trend_SMA_{self.meddle}", label=f"_Trend_SMA_{self.meddle}", linestyle='--', linewidth=1, color="blue", ax=axs[0], legend=False)
            sns.lineplot(_X, x="Date", y=f"_Trend_SMA_{self.long}", label=f"_Trend_SMA_{self.long}", linestyle='--', linewidth=1, color="red", ax=axs[0], legend=False)
            axs[0].set_xlabel('Date')
            axs[0].set_ylabel('Price')

            sns.lineplot(_X, x="Date", y="Trend", label="Trend", linestyle='-', linewidth=2, color="black", alpha=1, ax=axs[1], legend=False)
            axs[1].set_ylabel('Uptrand')
            axs[1].set_ylabel('Trend', color='r')
            axs[1].tick_params(axis='y')

            _y = [-1, 0, 1]
            _labels = {-1: "Down", 0: 'Neutral', 1: 'Up'}
            axs[1].set_yticks(_y)
            axs[1].set_yticklabels([_labels[i] for i in _y])

            lines1, labels1 = axs[0].get_legend_handles_labels()
            lines2, labels2 = axs[1].get_legend_handles_labels()
            axs[0].legend(lines1 + lines2, labels1 + labels2, loc='upper left')

            for _, group_data in _X.groupby('_Trend_Group'):
                if group_data['Trend'].iloc[0] == 0:
                    continue

                axs[0].scatter(group_data.index, group_data['_Max'], color='green', label='Máximo Local', marker='v', s=100)
                axs[0].scatter(group_data.index, group_data['_Min'], color='red', label='Mínimo Local', marker='^', s=100)

                min_idx = group_data['_Close'].idxmin()
                max_idx = group_data['_Close'].idxmax()

                if group_data['Trend'].iloc[0] == 1:
                    axs[0].fill_between(group_data.index, group_data['_Close'].min(), group_data['_Close'].max(), color="green", alpha=0.2)

                if group_data['Trend'].iloc[0] == -1:
                    axs[0].fill_between(group_data.index, group_data['_Close'].min(), group_data['_Close'].max(), color="red", alpha=0.2)

            fig.suptitle('Trend Graphic (target)', fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.98])

            plt.savefig(self.path)
            plt.close()

        return _X
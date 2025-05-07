import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin

sns.set_theme(style='ticks')

class OnBalanceVolume(BaseEstimator, TransformerMixin):

    def __init__(self, graphic=True, path="../figures/OBV_Graphic.png"):
        self.graphic = graphic
        self.path = path

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        _X = X.copy()

        _X['OBV'] = (_X['_Volume'] * ((_X['_Close'] - _X['_Close'].shift(1)) > 0)).cumsum()

        if self.graphic:
            fig, ax = plt.subplots(figsize=(15, 5), dpi=100)

            # Plot OBV
            sns.lineplot(data=_X, x="Date", y="OBV", label="OBV", color="blue", ax=ax)

            ax.set_xlabel('Date')
            ax.set_ylabel('OBV')
            ax.set_title('On-Balance Volume (OBV)')
            ax.legend()

            plt.tight_layout()
            plt.savefig(self.path)
            plt.close()

        return _X

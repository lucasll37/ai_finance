import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin

sns.set_theme(style='ticks')

class IchimokuClouds(BaseEstimator, TransformerMixin):

    def __init__(self, window1=9, window2=26, window3=52, graphic=True, path="../figures/Ichimoku_Clouds_Graphic.png"):
        self.window1 = window1
        self.window2 = window2
        self.window3 = window3
        self.graphic = graphic
        self.path = path

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        _X = X.copy()

        # Calculando as médias móveis
        _X['_Ichimoku_Conversion_Line'] = (_X['_High'].rolling(window=self.window1).max() + _X['_Low'].rolling(window=self.window1).min()) / 2
        _X['_Ichimoku_Base_Line'] = (_X['_High'].rolling(window=self.window2).max() + _X['_Low'].rolling(window=self.window2).min()) / 2
        _X['_Ichimoku_Leading_Span_A'] = (_X['_Ichimoku_Conversion_Line'] + _X['_Ichimoku_Base_Line']) / 2
        _X['_Ichimoku_Leading_Span_B'] = (_X['_High'].rolling(window=self.window3).max() + _X['_Low'].rolling(window=self.window3).min()) / 2
        _X['Ichimoku'] = _X["_Ichimoku_Conversion_Line"] - _X["_Ichimoku_Base_Line"]

        if self.graphic:
            fig, ax = plt.subplots(figsize=(15, 5), dpi=100)

            # Plotando as Nuvens de Ichimoku
            sns.lineplot(data=_X, x="Date", y="_Close", label="Close", color="black", ax=ax)
            sns.lineplot(data=_X, x="Date", y="_Ichimoku_Conversion_Line", label="Conversion Line", color="blue", ax=ax)
            sns.lineplot(data=_X, x="Date", y="_Ichimoku_Base_Line", label="Base Line", color="red", ax=ax)
            sns.lineplot(data=_X, x="Date", y="_Ichimoku_Leading_Span_A", label="Leading Span A", color="green", ax=ax)
            sns.lineplot(data=_X, x="Date", y="_Ichimoku_Leading_Span_B", label="Leading Span B", color="orange", ax=ax)

            ax.fill_between(_X.index, _X["_Ichimoku_Leading_Span_A"], _X["_Ichimoku_Leading_Span_B"], where=_X["_Ichimoku_Leading_Span_A"] >= _X["_Ichimoku_Leading_Span_B"], color='lightgreen', alpha=0.3)
            ax.fill_between(_X.index, _X["_Ichimoku_Leading_Span_A"], _X["_Ichimoku_Leading_Span_B"], where=_X["_Ichimoku_Leading_Span_A"] < _X["_Ichimoku_Leading_Span_B"], color='lightcoral', alpha=0.3)

            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.set_title('Ichimoku Clouds')
            ax.legend()

            plt.tight_layout()
            plt.savefig(self.path)
            plt.close()

        return _X

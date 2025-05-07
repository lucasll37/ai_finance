import yfinance as yf
from pandas import to_datetime

def get_ticket(ticket = "VALE3.SA"):
    vale = yf.Ticker(ticket)
    data = vale.history(period="max")
    
    data.index = to_datetime(data.index, utc=False)
    data.index = data.index.tz_localize(None)
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    data.rename(columns={
        'Open': '_Open',
        'High': '_High',
        'Low': '_Low',
        'Close': '_Close',
        'Volume': '_Volume'
    }, inplace=True)

    return data
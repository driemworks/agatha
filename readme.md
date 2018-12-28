# MYNT
MYNT is a tool to help you predict future close prices for any given stock tickers.
![alt-text](https://github.com/driemworks/mynt/blob/master/resources/images/NTDOY_12-16-2018_lookback=101_epochs=100_batch_size=32.png?raw=true)

## Should I have faith in the predictions?
Absolutely not. But it might help you learn how to use keras.

## How it works
Mynt uses an LSTM network to predict close prices for a user-specified number of days in the future. The training data is downloaded via [Alpha Vantage](https://www.alphavantage.co/).

## Usage
Refer to [app.py](https://github.com/driemworks/mynt/blob/master/app.py), where you can set parameters that are used to train the network: tickers, epochs, batch size, look back, and number of days to predict, the alpha vantage api key.

## Future Enhancements
- Allow multiple data sources, including for cryptocurrencies (only alphavantage at the moment)
- Any suggestions?

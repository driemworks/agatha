# MYNT
![alt-text](https://img.shields.io/hexpm/l/plug.svg)
<pre>

Mynt is a tool to help you predict future close prices for any given stock tickers.
</pre>
![alt-text](https://github.com/driemworks/mynt/blob/master/resources/images/NTDOY_12-16-2018_lookback=101_epochs=100_batch_size=32.png?raw=true)

## Should I have faith in the predictions?
Absolutely not. 

## How it works
Mynt uses an LSTM network to predict close prices for a user-specified number of days in the future. The training data is downloaded via [Alpha Vantage](https://www.alphavantage.co/).

## Usage
First get an API key from Alpha Vantage. 
To train a model for a  particular ticker
``` python
model = getOrTrainModel(alpha_vantage_api_key, 'GE', scaler, epochs=100, look_back=look_back)
```
To predict future close prices for a stock
```
prediction = predictFuture(model, num_days_to_predict, show_plot, scaler, look_back=look_back)
```

Refer to [app.py](https://github.com/driemworks/mynt/blob/master/examples/app.py), for a working example.

## Future Enhancements
- Allow multiple data sources, including for cryptocurrencies (only alphavantage at the moment)
- Any suggestions? 
  

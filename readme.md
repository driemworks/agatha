# MYNT
![alt-text](https://img.shields.io/hexpm/l/plug.svg)  
Mynt is a tool to help you predict future close prices for any given stock tickers.

<p align="center">
<img src="https://github.com/driemworks/mynt/blob/master/resources/images/full.PNG?raw=true" alt="full" width="400" height="400">|<img src="https://github.com/driemworks/mynt/blob/master/resources/images/prediction.PNG?raw=true" alt="full" width="400" height="400">
</p>

## Should I have faith in the predictions?
Probably not. 

## How it works
Mynt uses an LSTM network to predict close prices for a user-specified number of days in the future. The training data is downloaded via [Alpha Vantage](https://www.alphavantage.co/).

## Requirements
- python 3.5 or higher

## Installation
Clone this repository. Inside the Mynt folder, create the mynt package using
``` python
python setup.py sdist
```
Then install using pip. If you are using windows
``` python
pip install ./dist/Mynt-1.0-dev.zip
```
If you are using a unix based OS
``` python
pip install ./dist/Mynt-1.0-dev.tar.gz
```
If you use anaconda, you can load the conda environment using the environment.yml file in `resources/conda`
and running ```conda env create -f environment.yml```

## Usage
First get an API key from Alpha Vantage. 
To train a model for a  particular ticker, use
``` python
model = getOrTrainModel(alpha_vantage_api_key, ticker, alphavantage_data,
						model_data, weights_data, epochs=epochs, look_back=look_back)
```
- alphavantage data is downloaded as a csv and then pickled (saved as .pkl)
- the model_data is saved as json
- the weights file is saved as .h5

Predictions for future close prices for a stock can have output type as `json` or `plot` (will show pyplot)  
``` python
prediction_output = predictFuture(model, num_days_to_predict, ouptut_type)
```
Ex JSON output:
```
[
   {
      "price":"8.105009",
      "day":"1"
   },
   {
      "price":"7.9884334",
      "day":"2"
   },
   ...
   }
]
```
Refer to [app.py](https://github.com/driemworks/mynt/blob/master/examples/app.py), for a working example.

## Future Enhancements
- Need to rename project (Any suggestions?)
  

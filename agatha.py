import AlphaVantageGateway as avg
import DataUtils as du
import numpy as np
import json
import NetworkUtils as nu
from sklearn.preprocessing import MinMaxScaler

# use this to normalize all values between 0 and 1 (to train the model)
scaler = MinMaxScaler(feature_range=(0, 1))

class ModelData():
	def __init__(self, ticker, model, train_x, test_x, train_y, test_y, dataset, look_back, column):
		self.ticker = ticker
		self.model = model
		self.train_x = train_x
		self.test_x = test_x
		self.train_y = train_y
		self.test_y = test_y
		self.dataset = dataset
		self.look_back = look_back
		self.column = column


def getOrTrainModel(api_key, ticker, alphavantage_data_path, attribute, model_path, weights_path,
					epochs=100, batch_size=32, look_back=31):
	'''
	Use the provided parameters to train a model (at the specified model_path), save the weights to the weights_path,
	and cache the price data from alpha vantage in the alphavantage_data_path.

	If cached data is found, then use the cached price data and weights to compile the trained model

	:param api_key: The alphavantage api key
	:param ticker: The stock ticker
	:param alphavantage_data_path: The path to alphavantage data
	:param attribute: The column to predict. This can be 'open', 'close', or 'volume'
	:param model_path: The path to a model, or path to where a model should be saved
	:param weights_path: The path to model weights, or path to where weights should be saved
	:param epochs: The epochs used to train the model
	:param batch_size: The batch_size used to train the model
	:param look_back: The look_back value used to train the model
	:return: The model
	'''
	dataframe = avg.get_alpha_vantage_data(api_key, ticker, alphavantage_data_path)
	train_x, test_x, train_y, test_y, dataset = du.prepareTrainingData(dataframe, attribute, scaler, look_back)
	# create and fit the LSTM network
	model = nu.getModel(train_x, train_y, test_x, test_y, model_path, weights_path,
						epochs=epochs, batch_size=batch_size, look_back=look_back)
	return ModelData(ticker, model, train_x, test_x, train_y, test_y, dataset, look_back, attribute)


def predictFuture(model_data, num_days_to_predict, output_type):
	'''
	Predict the model output num_days_to_predict days in the future
	:param model_data: The model_data holds the model, test data, training data, and look back value
	:param num_days_to_predict: The number of days to predict in the future
	:param output_type: The output type can be either 'plot' or 'json'
	:return: JSON if output type is 'json', show plot if output type is 'plot'
	'''
	# make predictions
	model = model_data.model
	train_predict = model.predict(model_data.train_x)
	test_predict = model.predict(model_data.test_x)
	future_predict = nu.predictFuture(model, np.asarray(model_data.test_x[-1:]), num_days_to_predict, scaler)

	train_predict, test_predict, train_y, test_y =\
		nu.invert_predictions(train_predict, test_predict, model_data.train_y, model_data.test_y, scaler)

	# nu.scorePrediction(train_predict, test_predict, train_y, test_y)
	if output_type == 'plot':
		du.plotData(model_data.dataset, model_data.look_back, train_predict, test_predict,
					np.asarray(future_predict), scaler, model_data.column + ' - ' + model_data.ticker)
		return
	elif output_type == 'json':
		return toJson(future_predict, model_data.ticker, model_data.column)
	else:
		return future_predict


def toJson(future_predict, ticker, column):
	'''
	Convert the future_predict list to json
	:param future_predict: The future predictions
	:return: the json representation of the future predictions
	'''
	json_strings = {}
	json_strings['ticker'] = ticker
	json_strings['column'] = column

	datas = []
	i = 1
	for future in future_predict:
		val = future[0]
		data = {}
		data['day'] = str(i)
		data['price'] = str(val)
		datas.append(data)
		i += 1

	json_strings['predictions'] = datas
	return json.dumps(json_strings)
import numpy as np

import DataUtils as du
import NetworkUtils as nu
import AlphaVantageGateway as avg


class ModelData():
	def __init__(self, ticker, model, train_x, test_x, train_y, test_y, dataset):
		self.ticker = ticker
		self.model = model
		self.train_x = train_x
		self.test_x = test_x
		self.train_y = train_y
		self.test_y = test_y
		self.dataset = dataset


def formatFilePaths(ticker, epochs, batch_size, look_back, alpha_vantage_data_path, model_path, weights_path):
	cache_path_formatted = alpha_vantage_data_path.format(ticker)
	model_path_formatted = model_path.format(ticker, epochs, batch_size, look_back)
	weights_path_formatted = weights_path.format(ticker, epochs, batch_size, look_back)
	return cache_path_formatted, model_path_formatted, weights_path_formatted


def getOrTrainModel(api_key, ticker, scaler, epochs=100, batch_size=32, look_back=31,
					alphavantage_data_path='../resources/prices/alpha_vantage/{}.pkl',
					model_path='../resources/models/stock-prediction/{}_epochs={}&batch_size={}_lookback={}_model.json',
					weights_path='../resources/weights/stock-prediction/{}_epochs={}&batch_size={}_lookback={}_model.h5'):
	cache_path, model_path_formatted, weights_path_formatted = formatFilePaths(ticker, epochs, batch_size, look_back, 
																			   alphavantage_data_path, model_path,
																			   weights_path)
	dataframe = avg.get_alpha_vantage_data(api_key, ticker, cache_path)
	train_x, test_x, train_y, test_y, dataset = du.prepareTrainingData(dataframe, 'close', scaler, look_back)
	# create and fit the LSTM network
	model = nu.getModel(train_x, train_y, test_x, test_y, model_path_formatted, weights_path_formatted,
						epochs=epochs, batch_size=batch_size, look_back=look_back)
	return ModelData(ticker, model, train_x, test_x, train_y, test_y, dataset)


def predictFuture(model_data, num_days_to_predict, plotOutput, scaler, look_back):
	# make predictions
	model = model_data.model
	train_predict = model.predict(model_data.train_x)
	test_predict = model.predict(model_data.test_x)
	future_predict = nu.predictFuture(model, np.asarray(model_data.test_x[-1:]), num_days_to_predict, scaler)

	train_predict, test_predict, train_y, test_y =\
		nu.invert_predictions(train_predict, test_predict, model_data.train_y, model_data.test_y, scaler)

	# nu.scorePrediction(train_predict, test_predict, train_y, test_y)
	if plotOutput:
		du.plotData(model_data.dataset, look_back, train_predict, test_predict,
					np.asarray(future_predict), scaler, model_data.ticker)

	return future_predict

import numpy as np
from sklearn.preprocessing import MinMaxScaler

import PricesGateway as pg
import DataUtils as du
import NetworkUtils as nu

def formatFilePaths(ticker, alphavantage_data_path, model_path, weights_path):
	cache_path_formatted = alphavantage_data_path.format(ticker)
	model_path_formatted = model_path.format(ticker)
	weights_path_formatted = weights_path.format(ticker)
	return cache_path_formatted, model_path_formatted, weights_path_formatted

def trainModelAndMakePredictions(tickers, alphavantage_data_path, model_path, weights_path, epochs=100, batch_size=32, look_back=31, num_days_to_predict=30, plotOutput=False):
	future_predictions = {}
	for ticker in tickers:
		cache_path, model_path_formatted, weights_path_formatted = formatFilePaths(ticker, alphavantage_data_path, model_path, weights_path)
		dataframe = pg.get_alphavantage_data(ticker, cache_path)
		scaler = MinMaxScaler(feature_range=(0, 1))
		trainX, testX, trainY, testY, dataset = du.prepareTrainingData(dataframe, 'close', scaler, look_back)

		# create and fit the LSTM network
		model = nu.getModel(trainX, trainY, testX, testY, model_path_formatted, weights_path_formatted,
							epochs=epochs, batch_size=batch_size, look_back=look_back)
		# make predictions
		trainPredict = model.predict(trainX)
		testPredict = model.predict(testX)
		futurePredict = nu.predictFuture(model, np.asarray(testX[-1:]), num_days_to_predict, scaler)

		trainPredict, testPredict, trainY, testY = nu.invert_predictions(trainPredict, testPredict, trainY, testY, scaler)
		nu.scorePrediction(trainPredict, testPredict, trainY, testY)
		if plotOutput:
			du.plotData(dataset, look_back, trainPredict, testPredict, np.asarray(futurePredict), scaler, ticker)

		future_predictions[ticker] = convertFuturePredictToDict(futurePredict)

	return future_predictions


def convertFuturePredictToDict(futurePredict):
	values = {}
	i = 0
	for prediction in futurePredict:
		value = prediction[0]
		values[str(i)] = value
		i += 1
	return values

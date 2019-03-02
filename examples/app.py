from agatha import getOrTrainModel, predictFuture
import os
import json
import pickle
import matplotlib.pyplot as plt
# from tensorflow.python.client import device_lib
# from keras import backend as K

# os.environ['CDA_DEVICE_ORDER'] = 'PCI_BS_ID'
# os.environ['CDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# stock tickers of stocks to predict price for
# you can get data for nearly any symbol from alphavantage, though it doesn't have everything
tickers = ['F', 'AAPL', 'GE']
# your alphavantage api key
alpha_vantage_api_key = '1YB64755JWVBCS3'
attribute = 'close'
# the number of days in the future you want to predict (generally should be less than look_back)
num_days_to_predict = 10
# number of epochs to train the model
epochs = 100
# batch size to train the model
batch_size = 16
# cache candidate stock symbols
candidates = []

# the look_back value tells the LSTM to look back from now to X number of days ago and then make a prediction
# higher look back cold create better results, but be careful:
#  i) if there isn't much data, then using a high look_back value can lead to bad results, since you can only start
#     at index look_back. i.e. if your look_back is 100, then you start training from the 100th data point
#  ii) Too large of a look back can also lead to yor model being trained to recognize patterns that do not
#     affect the stock price (for example, does last years trend necessarily affect tomorrow's outcome?)
look_back = 15

def readCandidatePlots():
	for filename in os.listdir('./candidates'):
		plt.close()
		ax = plt.subplot()
		file = open('./candidates/' + filename, 'rb')
		ax = pickle.load(file)
		plt.show()


def predictClosePrices():
	print('Found ' + str(len(tickers)) + ' tickers to check')
	idx = 0
	for ticker in tickers:
		print('START - ' + ticker + ' - ' + str(idx) + '/' + str(len(tickers)))
		idx += 1
		# path where data is stored (or read if exists)
		alphavantage_data = '{}.pkl'.format(ticker)
		# path where model is stored (or read if exists)
		model_data = 'epochs={}&batch_size={}_lookback={}&attribte={}_model.json'.format(
			epochs, batch_size, look_back, attribute)
		# path where weights are stored (or read if exists)
		weights_data = '{}_epochs={}&batch_size={}_lookback={}_weights.h5'.format(
			ticker, epochs, batch_size, look_back)

		try:
			model = getOrTrainModel(alpha_vantage_api_key, ticker, alphavantage_data, attribute,
									model_data, weights_data, epochs=epochs, look_back=look_back)
			prediction_output = predictFuture(model, num_days_to_predict, 'json')
			prediction_json = json.loads(prediction_output)
			print(str(prediction_json))

			if prediction_json["percent change"] > 1:
				print('found a candidate')
				predictFuture(model, num_days_to_predict, 'plot', True, False, './candidates/')
				candidates.append(ticker)
		except Exception as e:
			print('An error occurred for ticker ' + ticker)
			print(str(e))
			continue

		print('END - ' + ticker)
	if candidates is not None:
		print(candidates)
		print('found ' + str(len(candidates)) + ' candidates')
	else:
		print('No candidates were fond')


if __name__ == "__main__":
	predictClosePrices()
	readCandidatePlots()

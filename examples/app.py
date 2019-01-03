from agatha import getOrTrainModel, predictFuture

# the ticker of the stock that you want to predict
ticker = 'GE'
# the output type (can be json or plot)
ouptut_type = 'json'
# your alphavantage api key
alpha_vantage_api_key = 'YOUR-API-KEY-FROM-ALPHAVANTAGE'
# the data attribute you want to predict (open, close, or volume)
attribute = 'close'
# the number of days in the future you want to predict (generally should be less than look_back)
num_days_to_predict = 20
# number of epochs to train the model
epochs = 100
# batch size to train the model
batch_size = 32

# the look_back value tells our LSTM to look back from now to X number of days ago and then make a prediction
# higher look back could create better results, but be careful:
#  i) if there isn't much data, then using a high look_back value can lead to bad results, since you can only start
#     at index look_back. i.e. if you look_back is 100, then you start training from the 100th data point
#  ii) Too large of a look back can also lead to your model being trained to recognize patterns that do not
#     affect the stock price (for example, does last years trend necessarily affect tomorrow's outcome?)
look_back = 25

# path where data is stored (or read if exists)
alphavantage_data = 'resources/prices/{}.pkl'.format(ticker)
# path where model is stored (or read if exists)
model_data = 'resources/models/epochs={}&batch_size={}_lookback={}_model.json'.format(
	epochs, batch_size, look_back)
# path where weights are stored (or read if exists)
weights_data = 'resources/weights/{}_epochs={}&batch_size={}_lookback={}_weights.h5'.format(
	ticker, epochs, batch_size, look_back)


if __name__ == "__main__":
	model = getOrTrainModel(alpha_vantage_api_key, ticker, alphavantage_data, attribute,
							model_data, weights_data, epochs=epochs, look_back=look_back)
	prediction_output = predictFuture(model, num_days_to_predict, ouptut_type)

	print(prediction_output)

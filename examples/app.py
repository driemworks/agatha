from mynt import getOrTrainModel, predictFuture

ticker = 'GE'
ouptut_type = 'json'
alpha_vantage_api_key = '1YB64755JWVBCSU3'

num_days_to_predict = 20
epochs = 100
batch_size = 32
look_back = 25

alphavantage_data = 'resources/prices/{}.pkl'.format(ticker)
model_data = 'resources/models/epochs={}&batch_size={}_lookback={}_model.json'.format(
	epochs, batch_size, look_back)
weights_data = 'resources/weights/{}_epochs={}&batch_size={}_lookback={}_weights.h5'.format(
	ticker, epochs, batch_size, look_back)

model = getOrTrainModel(alpha_vantage_api_key, ticker, alphavantage_data,
						model_data, weights_data, epochs=epochs, look_back=look_back)
prediction_output = predictFuture(model, num_days_to_predict, ouptut_type)
print(prediction_output)
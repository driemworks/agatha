import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def create_dataset(dataset, look_back=1):
	data_x, data_y = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		data_x.append(a)
		data_y.append(dataset[i + look_back, 0])
	return np.array(data_x), np.array(data_y)


def prepareTrainingData(dataframe, attribute, scaler, look_back):
	# get the specified column and convert to float64
	dataset = dataframe.loc[:, [attribute]]
	dataset = dataset.astype('float64')
	# normalize the dataset
	dataset = scaler.fit_transform(dataset)
	# split into train and test sets
	train_size = int(len(dataset) * 0.67)
	test_size = len(dataset) - train_size
	train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
	train_x, train_y = create_dataset(train, look_back)
	test_x, test_y = create_dataset(test, look_back)
	print("train x shape " + str(train_x.shape))
	print("train y shape " + str(train_y.shape))
	# reshape input to be [samples, time steps, features]
	train_x = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
	test_x = np.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))
	return train_x, test_x, train_y, test_y, dataset


def plotData(dataset, look_back, trainPredict, testPredict, futurePredict, scaler, title, save, save_dir, show):
	ax = plt.subplot()

	testTerminalIndex = len(testPredict) + len(trainPredict) + 2*look_back

	if trainPredict is not None:
		# shift train predictions for plotting
		trainPredictPlot = np.empty_like(dataset)
		trainPredictPlot[:, :] = np.nan
		trainPredictPlot[look_back-1:len(trainPredict) + look_back - 1, :] = trainPredict
		ax.plot(trainPredictPlot, label='train')

	if testPredict is not None:
		# shift test predictions for plotting
		testPredictPlot = np.empty_like(dataset)
		testPredictPlot[:, :] = np.nan
		testPredictPlot[len(trainPredict) + (look_back * 2):testTerminalIndex :] = testPredict
		ax.plot(testPredictPlot, label='test')

	if futurePredict is not None:
		# shift future prediction for plotting
		num_days_predicted = len(futurePredict)
		dataset_length = len(dataset)
		futurePredictPlot = np.empty((num_days_predicted + dataset_length, 1))
		futurePredictPlot[:, :] = np.nan
		futurePredictPlot[testTerminalIndex + 1:testTerminalIndex + 1 + num_days_predicted, :] = futurePredict
		ax.plot(futurePredictPlot, label='prediction')
	# plot baseline
		ax.plot(scaler.inverse_transform(dataset), label='actual')
	plt.grid(color='black', linestyle='-', linewidth=0.5)
	plt.legend(loc=1, fontsize='small')
	plt.xlabel('Day')
	plt.ylabel('Price (USD)')
	plt.title(title)
	if save:
		file = open(save_dir + title + '.pkl', 'wb')
		pickle.dump(ax, file)
	if show:
		plt.show()


def prepareData(dataframe, scaler, look_back):
	# will only work for stock data at the moment
	dataset = dataframe.loc[:, ['adj_close']]
	print('dataset length: ' + str(len(dataset)))
	dataset = dataset.astype('float32')
	# normalize the dataset
	dataset = scaler.fit_transform(dataset)
	# return du.create_dataset(dataset, look_back)
	x, y = create_dataset(dataset, look_back)
	train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.67, shuffle=False)
	train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))
	test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))
	return train_x, train_y, test_x, test_y, dataset

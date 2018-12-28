from network import trainModelAndMakePredictions
import numpy as np
import NetworkUtils as nu
# fix random seed for reproducibility
np.random.seed(7)
# tickers = ['T', 'IBM', 'NFLX', 'HPE', 'AMD', 'AMRS', 'NVDA', 'SNAP', 'NTDOY', 'SNE']
# tickers = ['BAC', 'DAL', 'TWTR', 'FIT', 'GE', 'TWLO', 'AMD', 'SUN', 'FB', 'F', 'TSLA', 'AMZN']
# MRIN, TLGT, GTIM, MU, PVTL=> not enough data to test though
# ON, PS? => not enough data to test
# BAC, VOD, STI, NTDOY, SNE
# tickers = ['MU']
# tickers = ['TNDM', 'NFEC', 'CDNA','TPNL','AMSC','NIHD','IMMY','I', 'MDB', 'ECYT', 'LFVN', 'TWLO', 'SHSP', 'PRQR', 'CDXS', 'ABCD', 'INS', 'VCEL', 'ARWR', 'AYX', 'WWE', 'CATS', 'OSIR', 'AMED','OKTA']
tickers = ['MANT']
epochs = 500
batch_size = 32

look_back = 15

num_days_to_predict = 7

up_arrow = '▲'
down_arrow = '▼'

alphavantage_data_path = './resources/prices/alpha_vantage/{}.pkl'

model_path = "./resources/models/stock-prediction/{}_epochs={}&batch_size={}_lookback={}_model.json"\
	.format('{}', epochs, batch_size, look_back)
weights_path = "./resources/weights/stock-prediction/{}_epochs={}&batch_size={}_lookback={}_model.h5"\
	.format('{}', epochs, batch_size, look_back)

futures = trainModelAndMakePredictions(tickers, alphavantage_data_path, model_path, weights_path, epochs, batch_size, look_back, num_days_to_predict, True)

for ticker in tickers:
	futurePredict = futures[ticker]
	first_val = futurePredict[str(0)]
	last_val = futurePredict[str(len(futurePredict) - 1)]
	threshold = .5
	# now want to answer the question, should I buy, sell, or hold this stock?
	print(str(nu.determineAction(nu.calculateMomentum(last_val, first_val), threshold)))

print(str(futures))

# Resources
# https://ai.stackexchange.com/questions/3156/how-to-select-number-of-hidden-layers-and-number-of-memory-cells-in-lstm
# https://en.wikipedia.org/wiki/Momentum_(technical_analysis)
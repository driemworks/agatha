from flask import Flask, jsonify, request, app
import json
from Correlation.app import calculate_correlation
import pandas as pd


app = Flask(__name__)


@app.route('/correlation-matrix', methods=['GET'])
def getCorrelationCoefficients():
	r = request.get_json()
	# labels, corr = calculate_correlation(r['coins'])
	labels, corr = calculate_correlation(['ETH'])
	corr_collection = {}
	for i,j in enumerate(labels):
		inner_dict = {}
		for k, l in enumerate(labels):
			inner_dict[l] = pd.Series(corr[i][k]).to_json(orient='values')
		corr_collection[j] = inner_dict
	return json.dumps(corr_collection)


@app.route('/currencies', methods=['GET'])
def getAvailableCurrencies():
	currs = ['ETH', 'XMR', 'XRP']
	return jsonify(currs)

if __name__ == "__main__":
	app.run()



from mynt import getOrTrainModel, predictFuture
from sklearn.preprocessing import MinMaxScaler

num_days_to_predict = 5
show_plot = True
alpha_vantage_api_key = 'YOURAPIKEY'
look_back = 7

scaler = MinMaxScaler(feature_range=(0, 1))

model = getOrTrainModel(alpha_vantage_api_key, 'GE', scaler, epochs=100, look_back=look_back)
prediction = predictFuture(model, num_days_to_predict, show_plot, scaler, look_back=look_back)

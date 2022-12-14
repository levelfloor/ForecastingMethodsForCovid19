from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import r2_score
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from math import sqrt
# loading data
trainlength = int(input("Enter Train Data Length"))
laglength = int(input("Enter Lag Data Length"))
series = read_csv('COVDATA.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
# creating the lagged dataset
values = DataFrame(series.new_cases)
dataframe = concat([values.shift(1), values], axis=1)
dataframe.columns = ['t', 't+1']
# splitting into train and test datasets
X = dataframe.values
train_size = int(730)
train, test = X[1:trainlength], X[trainlength:]
#X is the first original dataset, and Y is the same but shifted one over
train_X, train_y = train[:,0], train[:,1]
test_X, test_y = test[:,0], test[:,1]
# persistence model on training set
#all x for which x is in range of train_X, isolating values?
train_pred = [x for x in train_X]
# calculate residuals
# calculates that off by one residual score
train_resid = [train_y[i]-train_pred[i] for i in range(len(train_pred))]
# model the training set residuals
window = laglength
model = AutoReg(train_resid, lags=laglength)
model_fit = model.fit()
coef = model_fit.params
# walk forward over time steps in test
# finding that best fit line and correcting it with residuals
history = train_resid[len(train_resid)-window:]
history = [history[i] for i in range(len(history))]
predictions = list()
for t in range(len(test_y)):
	# persistence
	yhat = test_X[t]
	error = test_y[t] - yhat
	# predict error
	length = len(history)
	lag = [history[i] for i in range(length-window,length)]
	pred_error = coef[0]
	for d in range(window):
		pred_error += coef[d+1] * lag[window-d-1]
	# correct the prediction
	yhat = yhat + pred_error
	predictions.append(yhat)
	history.append(error)
	print('predicted=%f, expected=%f' % (yhat, test_y[t]))
# error
rmse = sqrt(mean_squared_error(test_y, predictions))
print(r2_score(test_y,predictions))
print('Test RMSE: %.3f' % rmse)
# plot predicted error
#pyplot.plot(test_y)
#pyplot.plot(predictions, color='red')
pyplot.plot(train_y)
pyplot.plot([None for i in train_y] + [x for x in test_y])
pyplot.plot([None for i in train_y] + [x for x in predictions])
pyplot.legend(['Train Data', 'Test Data', 'Prediction'])
pyplot.show()
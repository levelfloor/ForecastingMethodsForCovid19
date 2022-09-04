from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from matplotlib import pyplot
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from math import sqrt
trainlength = int(input("Enter Train Data Length"))
series = read_csv('COVDATA.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
values = DataFrame(series.new_cases)
dataframe = concat([values.shift(1), values], axis=1)
X = dataframe.values
train, test = X[0:trainlength], X[trainlength:]
train_X, train_y = train[:,0], train[:,1]
test_X, test_y = test[:,0], test[:,1]
def model_persistence(x):
	return x
predictions = list()
for x in test_X:
	yhat = model_persistence(x)
	predictions.append(yhat)
rmse = sqrt(mean_squared_error(test_y, predictions))
print(r2_score(test_y,predictions))
print('Test RMSE: %.3f' % rmse)
pyplot.plot(train_y)
pyplot.plot([None for i in train_y] + [x for x in test_y])
pyplot.plot([None for i in train_y] + [x for x in predictions])
pyplot.show()
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
series = read_csv('COVDATA.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
values = DataFrame(series.new_cases)
Y = values
X = np.array(list(range(1,782)))
X=X.reshape(-1,1)
fig = plt.figure('Plot Data + Regression')
reg1 = LinearRegression().fit(X, Y)
ax1 = fig.add_subplot(111)
ax1.plot(X, Y, marker='x', c='b', label='data')
ax1.plot(X,reg1.predict(X), marker='o',c='g', label='linear r.')
print('Test RMSE: %.3f' % sqrt(mean_squared_error(X, reg1.predict(X))))
ax1.set_title('Data vs Regression')
ax1.legend(loc=2)
plt.show()

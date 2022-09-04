import pwlf
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
import numpy as np
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt

#fit your data (x and y)
series = read_csv('COVDATA.csv', header=0, index_col=0, parse_dates=True, squeeze=True)

myPWLF = pwlf.PiecewiseLinFit(np.array(list(range(1,782)))
, series['new_cases'])

#fit the data for n line segments
z = myPWLF.fit(5)
#5
#first shot 25
#second shot 50
#calculate slopes
slopes = myPWLF.calc_slopes()

# predict for the determined points
xHat = np.array(list(range(1,782)))

yHat = myPWLF.predict(xHat)

rmse = sqrt(mean_squared_error(series['new_cases'], yHat))
print(r2_score(series['new_cases'],yHat))
print('Test RMSE: %.3f' % rmse)

plt.figure()
plt.plot(np.array(list(range(1,782)))
, series['new_cases'])
plt.plot(xHat, yHat, '-')
plt.show()



#calculate statistics

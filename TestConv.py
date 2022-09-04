import pandas as pd
import matplotlib.pyplot as plt
from autots import AutoTS
import numpy as np
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
import plotly.express as px

#df = pd.read_csv("COVDATA.csv", usecols=['date', 'new_cases'])
df2 = pd.read_csv("COVDATA.csv")
#df['date'] = pd.to_datetime(df['date'])
#df = df.sort_values('date')
#train_df = df.iloc[:730]
#test_df = df.iloc[730:]
real_data = df2.iloc[:735]

#730
#train_df.new_cases.plot(figsize=(15,8), title= 'CovidData', fontsize=14, label='Train')
#test_df.new_cases.plot(figsize=(15,8), title= 'CovidData', fontsize=14, label='Test')
model = AutoTS(forecast_length=46, frequency='infer', ensemble='simple')
#79
model.fit(real_data, date_col="date", value_col='new_cases',  id_col=None)
future_predictions = model.predict()
forecast = future_predictions.forecast
forecast = pd.DataFrame(forecast, columns = ['new_cases']).to_csv('trueprediction14.csv')

data = pd.read_csv("trueprediction14.csv")
data.rename( columns={'Unnamed: 0':'date'}, inplace=True )
allData = pd.read_csv("COVDATA.csv")
#730 - 51
#700 - 81
#720 - 61
#710 - 71

print(data)
rmse = sqrt(mean_squared_error(allData['new_cases'][735:], data['new_cases']))
print(r2_score(allData['new_cases'][735:], data['new_cases']))
print('Test RMSE: %.3f' % rmse)
plt.plot(allData.date, allData.new_cases,linestyle='--', marker='.', color = 'blue', label='Full range')
plt.plot(data.date, data.new_cases,linestyle='--', marker='.', color = 'orange', label='Full range')
plt.gcf().autofmt_xdate()
plt.xticks(np.arange(0, len(allData)+1, 20))
plt.legend(['Real_Data', 'Prediction'])
plt.show()



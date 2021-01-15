import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2_contingency
from scipy.stats import f_oneway
from scipy.stats import spearmanr
from scipy.stats import shapiro
from scipy.stats import norm
from sklearn.preprocessing import LabelEncoder
from scipy.stats import pearsonr
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from matplotlib import pyplot
from pandas import to_datetime
from fbprophet import Prophet
from pandas import DataFrame
from fbprophet.plot import plot_plotly, plot_components_plotly
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt

def get_arima_result(data, column_1, column_2):
	df = pd.DataFrame(index= data.index, columns=['ds','y'])
	df['ds'] = to_datetime(data[column_1].values)
	df.index = df['ds'].values
	df['y'] = data[column_2].values
	df_new = pd.DataFrame(index=df.index, columns=['y'])
	df_new['y'] = df['y'].values
	df_new_array = df_new.values

	size = int(len(df_new_array) * 0.99)
	train, test = df_new_array[0:size], df_new_array[size:len(df_new_array)]
	history = [x for x in train]
	predictions = list()

	# walk-forward validation
	for t in range(len(test)):
		model = ARIMA(history, order=(1, 1, 0))
		model_fit = model.fit()
		output = model_fit.forecast()
		yhat = output[0]
		predictions.append(yhat)
		obs = test[t]
		history.append(obs)
		print('predicted=%f, expected=%f' % (yhat, obs))

	# evaluate forecasts
	rmse = sqrt(mean_squared_error(test, predictions))
	print('Test RMSE: %.3f' % rmse)
	# plot forecasts against actual outcomes
	pyplot.plot(test)
	pyplot.plot(predictions, color='red')
	pyplot.show()

data = pd.read_csv('/home/tansen/my_files/dataScienceLab/latest.csv')
data = data.query('country_name in ["France","Germany","United Kingdom"]')
data = data[data['date'] > '2020-03-20']
data = data.groupby("date")[['Deaths_New','Recovered_New', 'Confirmed_New','Deaths','Confirmed', 'Recovered']].sum()
#data = data.groupby("date")[['Confirmed', 'Recovered']].sum()
data = data[data['Deaths'] > 0]
data['date'] = data.index
index = [i for i in range(len(data))]
data.index = index

data['Deaths'] = data['Deaths']
data['Deaths'] = data['Deaths'].apply(np.log)
data['Confirmed'] = data['Confirmed']
data['Confirmed'] = data['Confirmed'].apply(np.log)

fartality_rate_column = [data.loc[i, 'Deaths']/data.loc[i,'Confirmed']
						 if data.loc[i,'Deaths']!=0 else 0 for i in data.index]


def adfuller_test(sales):
    result=adfuller(sales)
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations']
    for value,label in zip(result,labels):
        print(label+' : '+str(value) )

    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data is stationary")
    else:
        print("weak evidence against null hypothesis,indicating it is non-stationary ")

fartality_rate_column = data['Deaths'] / data['Confirmed']
data['fartality_rate'] = fartality_rate_column.values
adfuller_test(data['fartality_rate'])
data['fartality_rate'] = fartality_rate_column
#data['fartality_rate'] = data['fartality_rate'] + 1
#data['fartalityy_rate'] = data['fartality_rate'].apply(np.log)
get_arima_result(data, 'date', 'fartality_rate')
#data['fartality_rate'] = data['fartality_rate'] - data['fartality_rate'].shift(1)
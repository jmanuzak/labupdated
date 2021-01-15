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
from matplotlib import pyplot
from pandas import to_datetime
from fbprophet import Prophet
from pandas import DataFrame
from fbprophet.plot import plot_plotly, plot_components_plotly
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import datetime
from fbprophet.diagnostics import cross_validation,  performance_metrics

def get_prophet_result(train, test ,column_1, column_2, month_lim, day_lim ,date_format = '2020-%02d-%02d'):
	df_train = pd.DataFrame(index= train.index, columns=['ds','y'])
	#print(df)
	#exit()
	df_train['ds'] = to_datetime(train[column_1].values)
	df_train['y'] = train[column_2].values
	# print(df_train.tail())
	# exit()
	df_test = pd.DataFrame(index= test.index, columns=['ds','y'])
	#print(df)
	#exit()
	df_test['ds'] = to_datetime(test[column_1].values)
	#print(df)
	#exit()
	df_test['y'] = test[column_2].values

	# print(df_test.tail())
	# exit()
	model = Prophet()
	# fit the model
	model.fit(df_train)
	#exit()
	# define the period for which we want a prediction
	future = list()
	for i in range(1, month_lim):

		for j in range(1, day_lim):
			date = date_format % (i,j)
			future.append([date])
	#exit()
	# future.columns = ['ds']
	# future['ds']= to_datetime(future['ds'])
	future = DataFrame(future)
	#print(future)
	# exit()
	future.columns = ['ds']
	future['ds']= to_datetime(future['ds'])
	# print(future)
	# exit()

	forecast = model.predict(future)
	#exit()
	model.plot(forecast)
	model.plot_components(forecast)
	plot_plotly(model, forecast)
	plot_components_plotly(model, forecast)
	pyplot.show()
	# create test dataset, remove last 12 months
	#train = df.drop(df.index[-12:])

	#print(train)
	# exit()
	# calculate MAE between expected and predicted values for december
	#print(type(forecast['ds']))
	#exit()
	#print('###################')
	#print(df['ds'])
	#print('###################')
	#print(forecast['ds'])
	#exit()
	values_to_match = forecast['ds'].values
	filtered_test_df = df_test.loc[df_test['ds'].isin(values_to_match)]
	values_to_match_forcast = filtered_test_df['ds'].values
	forecast_test_df = forecast.loc[forecast['ds'].isin(values_to_match_forcast)]
	#print(y_true_filtered)
	#exit()
	y_true = filtered_test_df['y'].values
	y_pred = forecast_test_df['yhat'].values
	mae = mean_absolute_error(y_true, y_pred)

	print('MAE: %.3f' % mae)

	# plot expected vs actual
	pyplot.plot(y_true, label='Actual')
	pyplot.plot(y_pred, label='Predicted')
	date_index = [str(i)[:10] for i in filtered_test_df['ds'].values]
	#pyplot.xticks(np.arange(min(x), max(x)+1, 1.0))
	#dictionary = {k: v for v, k in enumerate(date_index)}
	#print(dictionary)
	#exit()
	#keys = np.arange(dictionary[min(dictionary, key=dictionary.get)], dictionary[max(dictionary, key=dictionary.get)] + 1, 5.0)
	#new_dicts = {v: k for k, v in dictionary.items()}
	#print(new_dicts)
	#d = [new_dicts[x] for x in keys]
	#print(d)
	#exit()
	pyplot.xticks(np.arange(len(date_index)), date_index )
	pyplot.legend()
	pyplot.show()
	# diagonistic tool
	# cv_results = cross_validation(model=model, initial="72 days", period="4 days", horizon="36 days")
	# cv_results
	# df_p = performance_metrics(cv_results)
	# df_p

data = pd.read_csv('/home/tansen/my_files/dataScienceLab/latest.csv')
data = data.query('country_name in ["France","Germany","United Kingdom"]')
data = data.groupby("date")[['Deaths_New','Recovered_New', 'Confirmed_New','Deaths','Confirmed', 'Recovered']].sum()
data['date'] = data.index
index = [i for i in range(len(data))]
data.index = index


data['Deaths'] = data['Deaths'] + 1
data['Deaths'] = data['Deaths'].apply(np.log)
data['Confirmed'] = data['Confirmed'] + 1
data['Confirmed'] = data['Confirmed'].apply(np.log)
# get_prophet_result(data, 'date', 'Deaths', 13)
#exit()
fartality_rate_column = [data.loc[i, 'Deaths']/data.loc[i,'Confirmed']
						 if data.loc[i,'Deaths']!=0 else 0 for i in data.index]

data['fartality_rate'] = fartality_rate_column
data['fartality_rate'] = data['fartality_rate'] + 1
data['fartality_rate'] = data['fartality_rate'].apply(np.log)

def sample_first_prows(data, perc=0.80):
    return data.head(int(len(data)*(perc)))

train = sample_first_prows(data)
test = data.iloc[max(train.index):]

get_prophet_result(train, test ,'date', 'fartality_rate', 11, 25)


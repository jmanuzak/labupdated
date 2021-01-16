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

data = pd.read_excel('/home/tansen/my_files/labupdated/First_Iteration.xlsx')
data = data.query('country_name in ["Germany","France","United Kingdom"]')
print(data.head(10))
#data = data.query('country_name in ["India"]')


columns = ['Deaths','date']
df = pd.DataFrame(data, columns=columns)
df['ds'] = to_datetime(df['date'])
df['ds'] = df['date']
df['y'] = df['Deaths']
# df = df[df['ds']>'2020-03-10']
# print(df[df['y'] < 0].head())


df.drop('Deaths', axis=1, inplace=True)
df.drop('date', axis=1, inplace=True)

daily_cases = df.sum(axis=0)
daily_cases.head()
daily_cases.plot()
plt.show()
model = Prophet()
# fit the model
model.fit(df)
# define the period for which we want a prediction
future = list()
for i in range(1, 13):
	date = '2021-%02d' % i
	future.append([date])
future = DataFrame(future)
future.columns = ['ds']
future['ds']= to_datetime(future['ds'])
# use the model to make a forecast
forecast = model.predict(future)
# summarize the forecast
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())
# plot forecast

model.plot(forecast)

model.plot_components(forecast)

plot_plotly(model, forecast)

plot_components_plotly(model, forecast)

pyplot.show()



# create test dataset, remove last 12 months
train = df.drop(df.index[-12:])
print(train.tail())

# calculate MAE between expected and predicted values for december
y_true = df['y'][-12:].values
y_pred = forecast['yhat'].values
mae = mean_absolute_error(y_true, y_pred)
print('MAE: %.3f' % mae)

# plot expected vs actual
pyplot.plot(y_true, label='Actual')
pyplot.plot(y_pred, label='Predicted')
pyplot.legend()
pyplot.show()
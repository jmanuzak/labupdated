import pandas as pd
from fbprophet import Prophet
from pandas import to_datetime
import matplotlib.pyplot as plt
from fbprophet import Prophet

plt.style.use('fivethirtyeight')
data = pd.read_csv('/home/tansen/my_files/dataScienceLab/latest.csv')

# for col in data.columns:
#     print(col)

data = data.query('country_name in ["Germany","France","United Kingdom"]')

#for prediction varialbes
#new_confirmed,new_death,new_recovered

columns = ['Deaths','date']
df = pd.DataFrame(data, columns=columns)
df = df.rename(columns={'date': 'ds',
                        'Deaths': 'y'})

# df = df[df['ds']>'2020-03-10']
print(df.head())
df['ds'] = pd.DatetimeIndex(df['ds'])

ax = df.set_index('ds').plot(figsize=(12, 8))
ax.set_ylabel('number of Deaths')
ax.set_xlabel('Date')

plt.show()

#######prophet for predict the future ##############
my_model = Prophet(interval_width=0.95)
my_model.fit(df)

future_dates = my_model.make_future_dataframe(periods=12, freq='MS')
future_dates.tail()

forecast = my_model.predict(future_dates)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
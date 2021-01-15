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

data = pd.read_csv('/home/tansen/my files/dataScienceLab/refined.csv')
data = data.query('country_name in ["Germany","France","United Kingdom"]')

# data['year'] = pd.DatetimeIndex(data['date']).year
# data['month'] = pd.DatetimeIndex(data['date']).month
# data['day'] = pd.DatetimeIndex(data['date']).day
#
#
# #######################################

X = data[['Deaths','Recovered']]
# X = data


y = data['total_confirmed'].values
#total_confirmed is highly importance for testing_policy 70%


# X = data[["school_closing" , "workplace_closing", "cancel_public_events",
# 		  "restrictions_on_gatherings", "public_transport_closing",
# 		  "stay_at_home_requirements"
# 		  ]]

# y = data['total_confirmed'].values

#workplace_closing has 50% impact on total_confirmed


# X = data[["Recovered", "investment_in_vaccines",
# 		  "emergency_investment_in_healthcare", "public_transport_closing",
# 		  "stay_at_home_requirements", "gdp", "nurses" , "physicians" ,"life_expectancy", "population_density"
# 		  ]]

# y = data['Deaths'].values

names = X.columns

sc = MinMaxScaler()
X = X.iloc[:,:].values
y = y.reshape(len(y),1)
X = sc.fit_transform(X)
y = sc.fit_transform(y)

model = RandomForestRegressor()
# fit the model
model.fit(X, y)

# get importance
importance = model.feature_importances_

# summarize feature importance
for i,v in enumerate(importance):
	#print('Feature: %0d, Score: %.5f' % (i,v))
	print(names[i], 'has importance', v)



# plot feature importance
plt.bar([x for x in range(len(importance))], importance)
plt.xticks(np.arange(2), names)
plt.show()


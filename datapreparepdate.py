import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np


data = pd.read_csv('/home/tansen/my_files/dataScienceLab/Covid_Week3.csv')
df = data

df['year'] = pd.DatetimeIndex(data['date']).year
df['month'] = pd.DatetimeIndex(data['date']).month
df['day'] = pd.DatetimeIndex(data['date']).day


totalnull_df = df.isnull().sum()
percentnull = totalnull_df/len(df)*100
print(percentnull)

main_data=df
print(data.columns)

print(data.shape)


# plt.scatter(data.nurses, data.health_expenditure)
# plt.show()

# plt.scatter(data.out_of_pocket_health_expenditure, data.health_expenditure)
# plt.show()

# plt.scatter(data.physicians, data.health_expenditure)
# plt.show()

#all relation for each other
sns.pairplot(data)
#for Histrogram
data.hist(figsize=(16,20), bins=50, xlabelsize=8, ylabelsize=8)
#coorelation
mask = np.tril(data.corr())
sns.heatmap(data.corr(), fmt='.1g', annot = True, cmap= 'cool', mask=mask)
###corelation
corr = data.drop('month',axis=1).corr()
sns.heatmap(corr[(corr>0.5) | (corr <= -0.4)],cmap='viridis',vmax=1.0,vmin=-1.0,linewidth=0.1,annot=True,annot_kws={"size":8},square=True)

#####check missing values

totalnull_df = df.isnull().sum()
percentnull = totalnull_df/len(df)*100
display(percentnull)
######density

data.plot(kind='density', subplots=True, layout=(4,4), sharex=False, figsize=(20, 20))

##########check month groupby

print(data.month.unique())

#######boxplot for nurses and check the outlier###################

sns.boxplot(y = df['nurses'])

ax = sns.barplot(x = df['nurses'], y="health_expenditure", data=data)
#ax = sns.barplot(x = df['nurses'], y="month", data=data)

print(df[df.nurses>15].country_name.unique()) # got ['Belgium' 'Switzerland' 'Ireland' 'Iceland' 'Norway'] for outlier

#####check for physican
sns.boxplot(y = df['physicians'])

ax = sns.barplot(x = df['physicians'], y="health_expenditure", data=data)
#ax = sns.barplot(x = df['physicians'], y="month", data=data)


print(df[df.nurses>6].country_name.unique()) #  got outlier for ['Austria' 'Australia' 'Belgium' 'Brazil' 'Belarus' 'Canada' 'Switzerland'
 # 'Chile' 'Cuba' 'Czech Republic' 'Germany' 'Denmark' 'Estonia' 'Finland'
 # 'France' 'United Kingdom' 'Croatia' 'Ireland' 'Iceland' 'Japan'
 # 'South Korea' 'Kuwait' 'Lithuania' 'Luxembourg' 'Netherlands' 'Norway'
 # 'New Zealand' 'Poland' 'Portugal' 'Qatar' 'Romania' 'Serbia' 'Russia'
 # 'Sweden' 'Singapore' 'Slovenia' 'Slovakia' 'United States of America']

sns.boxplot(y = df['out_of_pocket_health_expenditure'])

ax = sns.barplot(x = df['out_of_pocket_health_expenditure'], y="health_expenditure", data=data)
# ax = sns.barplot(x = df['out_of_pocket_health_expenditure'], y="month", data=data)


print(df[df.out_of_pocket_health_expenditure>1000].country_name.unique()) # got ['Switzerland' 'Iceland' 'Norway' 'United States of America'] as outlier


sns.boxplot(y = df['health_expenditure'])

# ax = sns.barplot(x = df['health_expenditure'], y="month", data=data)

print(df[df.health_expenditure>3500].country_name.unique())
#got outlier for ['Austria' 'Australia' 'Belgium' 'Canada' 'Switzerland' 'Germany'
 # 'Denmark' 'Finland' 'France' 'United Kingdom' 'Ireland' 'Iceland' 'Japan'
 # 'Luxembourg' 'Netherlands' 'Norway' 'New Zealand' 'Sweden'
 # 'United States of America']
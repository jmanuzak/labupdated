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
# data = pd.read_csv('/home/tansen/my files/dataScienceLab/Covid_Week3.csv')
# df = data
#
# for col in data.columns:
# 	print(col)
#
# data['year'] = pd.DatetimeIndex(data['date']).year
# data['month'] = pd.DatetimeIndex(data['date']).month
# data['day'] = pd.DatetimeIndex(data['date']).day
#
#
# #######################################

data = pd.read_csv('/home/tansen/my_files/dataScienceLab/refined.csv')
X = data[["total_confirmed" , "total_deceased" , "Recovered","urban_population",
 "rural_population"]]

y = data['population_density'].values

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
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()




#########################################


# check guisen distributed or not
stat, p = shapiro(data.population_density)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably Gaussian')
else:
	print('Probably not Gaussian')


#this section for new_confirm cases
#show scatter plot for new_confirm cases and population_density

plt.scatter(data.population_density, data.new_confirmed)
plt.show()

#see the country name alfter the plot

print(data[data['new_confirmed']>7900].country_name.unique())

#show boxplot
ax = sns.barplot( x = 'population_density', y = 'new_confirmed', data=data)

# Pearson’s Correlation Coefficient
# Tests whether two samples have a linear relationship

stat, p = pearsonr(data.population_density, data.new_confirmed)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')

# Spearman’s Rank Correlation
# Tests whether two samples have a monotonic relationship.
stat, p = spearmanr(data.population_density, data.new_confirmed)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')

# Analysis of Variance Test (ANOVA)
# Tests whether the means of two or more independent samples are significantly different.

stat, p = f_oneway(data.population_density, data.new_confirmed)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably the same distribution')
else:
	print('Probably different distributions')


#this section for new_deceased
#show scatter plot for new_confirm cases and population_density

plt.scatter(data.population_density, data.new_deceased)
plt.show()

#see the country name alfter the plot

print(data[data['new_deceased']>7900].country_name.unique())

#show boxplot
ax = sns.barplot( x = 'population_density', y = 'new_deceased', data=data)

# Pearson’s Correlation Coefficient
# Tests whether two samples have a linear relationship

stat, p = pearsonr(data.population_density, data.new_deceased)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')

# Spearman’s Rank Correlation
# Tests whether two samples have a monotonic relationship.
stat, p = spearmanr(data.population_density, data.new_confirmed)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')

# Analysis of Variance Test (ANOVA)
# Tests whether the means of two or more independent samples are significantly different.

stat, p = f_oneway(data.population_density, data.new_confirmed)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably the same distribution')
else:
	print('Probably different distributions')


 #this section for new_deceased
#show scatter plot for total_confirmed cases and population_density

plt.scatter(data.population_density, data.total_confirmed)
plt.show()


#show boxplot
ax = sns.barplot( x = 'population_density', y = 'total_confirmed', data=data)

# Pearson’s Correlation Coefficient
# Tests whether two samples have a linear relationship

stat, p = pearsonr(data.population_density, data.total_confirmed)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')

# Spearman’s Rank Correlation
# Tests whether two samples have a monotonic relationship.
stat, p = spearmanr(data.population_density, data.total_confirmed)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')

# Analysis of Variance Test (ANOVA)
# Tests whether the means of two or more independent samples are significantly different.

stat, p = f_oneway(data.population_density, data.total_confirmed)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably the same distribution')
else:
	print('Probably different distributions')
   

 #this section for new_deceased
#show scatter plot for population_male and population female cases and population_density

plt.scatter(data.population_density, data.total_confirmed)
plt.show()


#show boxplot
ax = sns.barplot( x = 'population_density', y = 'population_male', data=data)
ax = sns.barplot( x = 'population_density', y = 'population_female', data=data)

# Pearson’s Correlation Coefficient
# Tests whether two samples have a linear relationship

stat, p = pearsonr(data.population_density, data.population_male)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')

#for female 
stat, p = pearsonr(data.population_density, data.population_female)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')

# Spearman’s Rank Correlation
# Tests whether two samples have a monotonic relationship.
stat, p = spearmanr(data.population_density, data.population_male)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')

 #for female
stat, p = spearmanr(data.population_density, data.population_female)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')   

# Analysis of Variance Test (ANOVA)
# Tests whether the means of two or more independent samples are significantly different.

stat, p = f_oneway(data.population_density, data.population_male)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably the same distribution')
else:
	print('Probably different distributions')

#female
stat, p = f_oneway(data.population_density, data.population_female)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably the same distribution')
else:
	print('Probably different distributions')   

# Chi-Squared Test
# Tests whether two categorical variables are related or independent.
#for male 
table = [[data.population_density],[data.population_male]]
stat, p, dof, expected = chi2_contingency(table)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')

#for female 
table = [[data.population_density],[data.population_female]]
stat, p, dof, expected = chi2_contingency(table)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')


 #this section for new_deceased
#show scatter plot for rural_population 
# urban_population cases and population_density

plt.scatter(data.population_density, data.rural_population )
plt.show()


#show boxplot
# ax = sns.barplot( x = 'population_density', y = 'rural_population ', data=data)
# ax = sns.barplot( x = 'population_density', y = 'urban_population', data=data)

# Pearson’s Correlation Coefficient
# Tests whether two samples have a linear relationship

stat, p = pearsonr(data.population_density, data.rural_population)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')

#for urban 
stat, p = pearsonr(data.population_density, data.urban_population)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')

# Spearman’s Rank Correlation
# Tests whether two samples have a monotonic relationship.
stat, p = spearmanr(data.population_density, data.urban_population)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')

 #for female 
stat, p = spearmanr(data.population_density, data.rural_population)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')   

# Analysis of Variance Test (ANOVA)
# Tests whether the means of two or more independent samples are significantly different.

stat, p = f_oneway(data.population_density, data.rural_population)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably the same distribution')
else:
	print('Probably different distributions')

#female
stat, p = f_oneway(data.population_density, data.urban_population)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably the same distribution')
else:
	print('Probably different distributions')   

# Chi-Squared Test
# Tests whether two categorical variables are related or independent.
#for male 
table = [[data.population_density],[data.urban_population]]
stat, p, dof, expected = chi2_contingency(table)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')

#for female 
table = [[data.population_density],[data.rural_population]]
stat, p, dof, expected = chi2_contingency(table)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')

#*********************human development****************

plt.scatter(data.population_density, data.human_development_index )
plt.show()


# Pearson’s Correlation Coefficient
# Tests whether two samples have a linear relationship

stat, p = pearsonr(data.population_density, data.human_development_index)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')

# Spearman’s Rank Correlation
# Tests whether two samples have a monotonic relationship.
stat, p = spearmanr(data.population_density, data.human_development_index)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')
 

# Analysis of Variance Test (ANOVA)
# Tests whether the means of two or more independent samples are significantly different.

stat, p = f_oneway(data.population_density, data.human_development_index)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably the same distribution')
else:
	print('Probably different distributions')


#**************** open_street_maps ****************

plt.scatter(data.population_density, data.open_street_maps )
plt.show()


# Pearson’s Correlation Coefficient
# Tests whether two samples have a linear relationship

stat, p = pearsonr(data.population_density, data.open_street_maps)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')

# Spearman’s Rank Correlation
# Tests whether two samples have a monotonic relationship.
stat, p = spearmanr(data.population_density, data.open_street_maps)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')
 

# Analysis of Variance Test (ANOVA)
# Tests whether the means of two or more independent samples are significantly different.

stat, p = f_oneway(data.population_density, data.open_street_maps)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably the same distribution')
else:
	print('Probably different distributions')

#****************area*******************

plt.scatter(data.population_density, data.area )
plt.show()


# Pearson’s Correlation Coefficient
# Tests whether two samples have a linear relationship

stat, p = pearsonr(data.population_density, data.area)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')

# Spearman’s Rank Correlation
# Tests whether two samples have a monotonic relationship.
stat, p = spearmanr(data.population_density, data.area)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')
 

# Analysis of Variance Test (ANOVA)
# Tests whether the means of two or more independent samples are significantly different.

stat, p = f_oneway(data.population_density, data.open_street_maps)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably the same distribution')
else:
	print('Probably different distributions')

#****************life_expectancy*******************

plt.scatter(data.population_density, data.life_expectancy )
plt.show()


# Pearson’s Correlation Coefficient
# Tests whether two samples have a linear relationship

stat, p = pearsonr(data.population_density, data.life_expectancy)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')

# Spearman’s Rank Correlation
# Tests whether two samples have a monotonic relationship.
stat, p = spearmanr(data.population_density, data.life_expectancy)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')
 

# Analysis of Variance Test (ANOVA)
# Tests whether the means of two or more independent samples are significantly different.

stat, p = f_oneway(data.population_density, data.life_expectancy)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably the same distribution')
else:
	print('Probably different distributions')

#*******************pollution_mortality_rate****************

plt.scatter(data.population_density, data.pollution_mortality_rate )
plt.show()


# Pearson’s Correlation Coefficient
# Tests whether two samples have a linear relationship

stat, p = pearsonr(data.population_density, data.pollution_mortality_rate)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')

# Spearman’s Rank Correlation
# Tests whether two samples have a monotonic relationship.
stat, p = spearmanr(data.population_density, data.pollution_mortality_rate)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')
 

# Analysis of Variance Test (ANOVA)
# Tests whether the means of two or more independent samples are significantly different.

stat, p = f_oneway(data.population_density, data.pollution_mortality_rate)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably the same distribution')
else:
	print('Probably different distributions')

#*******************nurses****************

plt.scatter(data.population_density, data.nurses )
plt.show()


# Pearson’s Correlation Coefficient
# Tests whether two samples have a linear relationship

stat, p = pearsonr(data.population_density, data.nurses)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')

# Spearman’s Rank Correlation
# Tests whether two samples have a monotonic relationship.
stat, p = spearmanr(data.population_density, data.nurses)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')
 

# Analysis of Variance Test (ANOVA)
# Tests whether the means of two or more independent samples are significantly different.

stat, p = f_oneway(data.population_density, data.nurses)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably the same distribution')
else:
	print('Probably different distributions')


#*******************physicians****************

plt.scatter(data.population_density, data.physicians )
plt.show()


# Pearson’s Correlation Coefficient
# Tests whether two samples have a linear relationship

stat, p = pearsonr(data.population_density, data.physicians)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')

# Spearman’s Rank Correlation
# Tests whether two samples have a monotonic relationship.
stat, p = spearmanr(data.population_density, data.physicians)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')
 

# Analysis of Variance Test (ANOVA)
# Tests whether the means of two or more independent samples are significantly different.

stat, p = f_oneway(data.population_density, data.physicians)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably the same distribution')
else:
	print('Probably different distributions')


#*******************school_closing****************

plt.scatter(data.population_density, data.school_closing )
plt.show()


# Pearson’s Correlation Coefficient
# Tests whether two samples have a linear relationship

stat, p = pearsonr(data.population_density, data.school_closing)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')

# Spearman’s Rank Correlation
# Tests whether two samples have a monotonic relationship.
stat, p = spearmanr(data.population_density, data.school_closing)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')
 

#*****************workplace_closing******************

plt.scatter(data.population_density, data.workplace_closing )
plt.show()


# Pearson’s Correlation Coefficient
# Tests whether two samples have a linear relationship

stat, p = pearsonr(data.population_density, data.workplace_closing)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')

# Spearman’s Rank Correlation
# Tests whether two samples have a monotonic relationship.
stat, p = spearmanr(data.population_density, data.workplace_closing)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')
 

#*****************cancel_public_events******************

plt.scatter(data.population_density, data.cancel_public_events )
plt.show()


# Pearson’s Correlation Coefficient
# Tests whether two samples have a linear relationship

stat, p = pearsonr(data.population_density, data.cancel_public_events)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')

# Spearman’s Rank Correlation
# Tests whether two samples have a monotonic relationship.
stat, p = spearmanr(data.population_density, data.cancel_public_events)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')
 

#*****************restrictions_on_gatherings******************

plt.scatter(data.population_density, data.restrictions_on_gatherings )
plt.show()


# Pearson’s Correlation Coefficient
# Tests whether two samples have a linear relationship

stat, p = pearsonr(data.population_density, data.restrictions_on_gatherings)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')

# Spearman’s Rank Correlation
# Tests whether two samples have a monotonic relationship.
stat, p = spearmanr(data.population_density, data.restrictions_on_gatherings)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')
 


#*****************public_transport_closing******************

plt.scatter(data.population_density, data.public_transport_closing )
plt.show()


# Pearson’s Correlation Coefficient
# Tests whether two samples have a linear relationship

stat, p = pearsonr(data.population_density, data.public_transport_closing)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')

# Spearman’s Rank Correlation
# Tests whether two samples have a monotonic relationship.
stat, p = spearmanr(data.population_density, data.public_transport_closing)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')
 

#*****************stay_at_home_requirements******************

plt.scatter(data.population_density, data.stay_at_home_requirements )
plt.show()


# Pearson’s Correlation Coefficient
# Tests whether two samples have a linear relationship

stat, p = pearsonr(data.population_density, data.stay_at_home_requirements)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')

# Spearman’s Rank Correlation
# Tests whether two samples have a monotonic relationship.
stat, p = spearmanr(data.population_density, data.stay_at_home_requirements)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')
 
#*****************restrictions_on_internal_movement******************

plt.scatter(data.population_density, data.restrictions_on_internal_movement )
plt.show()


# Pearson’s Correlation Coefficient
# Tests whether two samples have a linear relationship

stat, p = pearsonr(data.population_density, data.restrictions_on_internal_movement)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')

# Spearman’s Rank Correlation
# Tests whether two samples have a monotonic relationship.
stat, p = spearmanr(data.population_density, data.restrictions_on_internal_movement)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')
 


#*****************income_support******************

plt.scatter(data.population_density, data.income_support )
plt.show()


# Pearson’s Correlation Coefficient
# Tests whether two samples have a linear relationship

stat, p = pearsonr(data.population_density, data.income_support)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')


# Spearman’s Rank Correlation
# Tests whether two samples have a monotonic relationship.
stat, p = spearmanr(data.population_density, data.income_support)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')


#*****************international_support******************

plt.scatter(data.population_density, data.international_support )
plt.show()


# Pearson’s Correlation Coefficient
# Tests whether two samples have a linear relationship

stat, p = pearsonr(data.population_density, data.international_support)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')


# Spearman’s Rank Correlation
# Tests whether two samples have a monotonic relationship.
stat, p = spearmanr(data.population_density, data.international_support)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')



#*****************public_information_campaigns******************

plt.scatter(data.population_density, data.public_information_campaigns )
plt.show()


# Pearson’s Correlation Coefficient
# Tests whether two samples have a linear relationship

stat, p = pearsonr(data.population_density, data.public_information_campaigns)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')


# Spearman’s Rank Correlation
# Tests whether two samples have a monotonic relationship.
stat, p = spearmanr(data.population_density, data.public_information_campaigns)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')


#*****************testing_policy******************

plt.scatter(data.population_density, data.testing_policy )
plt.show()


# Pearson’s Correlation Coefficient
# Tests whether two samples have a linear relationship

stat, p = pearsonr(data.population_density, data.testing_policy)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')


# Spearman’s Rank Correlation
# Tests whether two samples have a monotonic relationship.
stat, p = spearmanr(data.population_density, data.testing_policy)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')


#*****************contact_tracing******************

plt.scatter(data.population_density, data.contact_tracing )
plt.show()


# Pearson’s Correlation Coefficient
# Tests whether two samples have a linear relationship

stat, p = pearsonr(data.population_density, data.contact_tracing)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')


# Spearman’s Rank Correlation
# Tests whether two samples have a monotonic relationship.
stat, p = spearmanr(data.population_density, data.contact_tracing)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')

#*****************emergency_investment_in_healthcare******************

plt.scatter(data.population_density, data.emergency_investment_in_healthcare )
plt.show()


# Pearson’s Correlation Coefficient
# Tests whether two samples have a linear relationship

stat, p = pearsonr(data.population_density, data.emergency_investment_in_healthcare)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')



#*****************investment_in_vaccines******************

plt.scatter(data.population_density, data.investment_in_vaccines )
plt.show()


# Pearson’s Correlation Coefficient
# Tests whether two samples have a linear relationship

stat, p = pearsonr(data.population_density, data.investment_in_vaccines)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')



# Spearman’s Rank Correlation
# Tests whether two samples have a monotonic relationship.
stat, p = spearmanr(data.population_density, data.investment_in_vaccines)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')


#*****************Confirmed******************

plt.scatter(data.population_density, data.Confirmed )
plt.show()


# Pearson’s Correlation Coefficient
# Tests whether two samples have a linear relationship

stat, p = pearsonr(data.population_density, data.Confirmed)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')



# Spearman’s Rank Correlation
# Tests whether two samples have a monotonic relationship.
stat, p = spearmanr(data.population_density, data.Confirmed)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')

#*****************Deaths******************

plt.scatter(data.population_density, data.Deaths )
plt.show()


# Pearson’s Correlation Coefficient
# Tests whether two samples have a linear relationship

stat, p = pearsonr(data.population_density, data.Deaths)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')



# Spearman’s Rank Correlation
# Tests whether two samples have a monotonic relationship.
stat, p = spearmanr(data.population_density, data.Deaths)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')

#*****************Recovered******************

plt.scatter(data.population_density, data.Recovered )
plt.show()


# Pearson’s Correlation Coefficient
# Tests whether two samples have a linear relationship

stat, p = pearsonr(data.population_density, data.Recovered)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')



# Spearman’s Rank Correlation
# Tests whether two samples have a monotonic relationship.
stat, p = spearmanr(data.population_density, data.Recovered)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')
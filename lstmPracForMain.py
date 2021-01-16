import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
#country = "Unites States of America"
country = 'United States of America'

# Total COVID confirmed cases
data = pd.read_csv(
    "/home/tansen/my_files/labupdated/latest.csv")
# df_confirmed.to_csv('global.csv')
# print(df_confirmed.country_name.unique())
data = data[data["country_name"] == country]
data['Deaths'] = data['Deaths'] + 1
data['Deaths'] = data['Deaths'].apply(np.log)
data['Confirmed'] = data['Confirmed'] + 1
data['Confirmed'] = data['Confirmed'].apply(np.log)

data.index = pd.to_datetime(data['date'])
data['mortality_rate'] = data['Deaths'] / data['Confirmed']
del data['date']
columns = ['mortality_rate']
df_confirmed_country = pd.DataFrame(data, columns=columns)

print("Total days in the dataset", len(df_confirmed_country))

# Use data until 14 days before as training
x = len(df_confirmed_country) - 90

train = df_confirmed_country.iloc[:x]
test = df_confirmed_country.iloc[x:]

##scale or normalize data as the data is too skewed
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(train)

train_scaled = scaler.transform(train)
test_scaled = scaler.transform(test)

## Use TimeSeriestrain_generator to generate data in sequences.
# Alternatively we can create our own sequences.
from keras.preprocessing.sequence import TimeseriesGenerator

# Sequence size has an impact on prediction, especially since COVID is unpredictable!
seq_size = 7  ## number of steps (lookback)
n_features = 1  ## number of features. This dataset is univariate so it is 1
train_generator = TimeseriesGenerator(train_scaled, train_scaled, length=seq_size, batch_size=1)
print("Total number of samples in the original training data = ", len(train))  # 271
print("Total number of samples in the generated data = ", len(train_generator))  # 264 with seq_size=7

# Check data shape from generator
x, y = train_generator[10]  # Check train_generator
# Takes 7 days as x and 8th day as y (for seq_size=7)

# Also generate test data
test_generator = TimeseriesGenerator(test_scaled, test_scaled, length=seq_size, batch_size=1)
print("Total number of samples in the original training data = ", len(test))  # 14 as we're using last 14 days for test
print("Total number of samples in the generated data = ", len(test_generator))  # 7
# Check data shape from generator
x, y = test_generator[0]

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation

# Define Model
model = Sequential()
model.add(LSTM(150, activation='relu', return_sequences=True, input_shape=(seq_size, n_features)))
model.add(LSTM(64, activation='relu'))
model.add(Dense(64))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

model.summary()
print('Train...')
##########################

history = model.fit_generator(train_generator,
                              validation_data=test_generator,
                              epochs=50, steps_per_epoch=10)

# plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# forecast
prediction = []  # Empty list to populate later with predictions

current_batch = train_scaled[-seq_size:]  # Final data points in train
current_batch = current_batch.reshape(1, seq_size, n_features)  # Reshape

## Predict future, beyond test dates
future = 7  # Days
for i in range(len(test) + future):
    current_pred = model.predict(current_batch)[0]
    prediction.append(current_pred)
    current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

### Inverse transform to before scaling so we get actual numbers
rescaled_prediction = scaler.inverse_transform(prediction)

time_series_array = test.index  # Get dates for test data

# Add new dates for the forecast period
for k in range(0, future):
    time_series_array = time_series_array.append(time_series_array[-1:] + pd.DateOffset(1))

# Create a dataframe to capture the forecast data
df_forecast = pd.DataFrame(columns=["actual_confirmed", "predicted"], index=time_series_array)

df_forecast.loc[:, "predicted"] = rescaled_prediction[:, 0]
df_forecast.loc[:, "actual_confirmed"] = test["mortality_rate"]

# Plot
df_forecast.plot(title="Predictions for next 7 days")







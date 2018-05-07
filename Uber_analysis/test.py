import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

# Read data
df = pd.read_csv("uber_4weeks.csv")

# filter data by location id
df_loc1 = df[df['locationID']==1]

# delete first rowas to adjust to surge mulitplier update period (3 min = 12 rows)
df_new = df_loc1.drop(df.index[[0,7,12,17,21,28,31,36,40]])

# make timestamp an index
df_new.index = pd.to_datetime(df_new['timestamp'])

# resample ETA and multiplier in 3min intervals calculating mean value inside each interval
y1 = df_new.expected_wait_time.resample('3min', how='mean') # resampled ETA
y2 = df_new.surge_multiplier.resample('3min', how='mean') # resampled surge multiplier

# get lag with cross-rorrelation
from scipy.signal import correlate
from scipy.stats import zscore

y2 = y2[~np.isnan(y2)]
y1 = y1[~np.isnan(y1)]

a_sig = y2#df_new['surge_multiplier'].head(200).values
b_sig = y1#df_new['wait_time_mean'].head(200).values

difa = np.diff(a_sig)
difaN = zscore(difa)
difb = np.diff(b_sig)
difbN = zscore(difb)

#lag = np.argmax(correlate(a_sig, b_sig,mode='same'))
#c_sig = np.roll(b_sig, shift=int(np.ceil(lag)))
#lag = np.argmax(correlate(difa[0:20], difb[0:20],mode='same'))
difbNrol = np.roll(difbN, shift=int(1))

# add_regressor here https://github.com/facebook/prophet/blob/v0.2/notebooks/seasonality_and_holiday_effects.ipynb 


plt.plot(difaN)
#plt.plot(difb[0:20])
plt.plot(difbNrol)
plt.show()

# LINEAR REGRESSION
# fit linear regression
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Split the data into training/testing sets
difbN_train = difbNrol[:5000]
difbN_test = difbNrol[5000:]

# Split the targets into training/testing sets
difaN_train = difaN[:5000]
difaN_test = difaN[5000:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(difbN_train[:,np.newaxis], difaN_train[:,np.newaxis])

# Make predictions using the testing set
difaN_pred = regr.predict(difbN_test[:,np.newaxis])

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(difaN_test, difaN_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(difaN_test, difaN_pred))

# Plot outputs
plt.scatter(difbN_test, difaN_test,  color='black')
plt.plot(difbN_test, difaN_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()

#plot prediciton
#plt.plot(difaN_test[0:50],color='yellow')
plt.plot(difaN_test,color='blue')
plt.plot(difaN_pred,color='green')
plt.show()
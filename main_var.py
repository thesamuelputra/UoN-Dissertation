# 1. Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import statsmodels
from math import sqrt
from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
pd.options.plotting.backend = "plotly"

# 2. Importing the dateset
df = pd.read_csv("dataset.csv",parse_dates=['Date'])
# Drop Unused Column Based on Ganger-Causality Tests (P<0.1)
df.drop(["sunsetTime", "sunriseTime","energy_median","energy_mean",
	"energy_max","energy_count","stdorToU","Acorn_grouped",
	"energy_std","energy_min","precipType","Acorn","windBearing",
	"cloudCover","windSpeed","pressure","visibility","moonPhase",
	"holiday","temperatureMax","temperatureMin"], axis = 1, inplace = True)
mask = df['LCLid'].isin(['MAC000002','MAC000606','MAC000096','MAC000050',
	'MAC005555','MAC000098','MAC000225','MAC000059',
	'MAC001201','MAC000584','MAC000055','MAC000101',
	'MAC000114','MAC000107','MAC000363','MAC000025',
	'MAC000006','MAC001600'])
df = df.loc[mask]
data = df.loc[df['LCLid'] == "MAC000055"]
data.pop("LCLid")
data = data.set_index('Date')

# 3. Create train-test split
train = data.iloc[:-80,:]
test = data.iloc[-80:,:]
forecasting_model = VAR(train)

# 4. Look for optimal lag order (AIC)
# results_aic = []
# for p in range(1,10):
#   results = forecasting_model.fit(p)
#   results_aic.append(results.aic)
# sns.set()
# plt.plot(list(np.arange(1,10,1)), results_aic)
# plt.xlabel("Order")
# plt.ylabel("AIC")
# plt.show()

# 5. Modelling and Forecasting
results = forecasting_model.fit(6)
print(results.summary())
lagged_values = train.values[-6:]
intervals = results.forecast_interval(y= lagged_values, steps=80, alpha=0.05)
forecast = pd.DataFrame(results.forecast(y= lagged_values, steps=80),
	index = test.index, columns= ['humidityPred', 'uvIndexPred', 'temperatureMeanPred', 'energy_sumPred'])



# forecast['humidity'] = data['humidity'].iloc[-80]
# forecast['uvIndex'] = data['uvIndex'].iloc[-80]
# forecast['temperatureMean'] = data['temperatureMean'].iloc[-80]
# forecast['energy_sum'] = data['energy_sum'].iloc[-80]
energyForecast = forecast[['energy_sumPred']]
energyForecast['Lower-CI'] = intervals[1][:,-1]
energyForecast['Upper-CI'] = intervals[-1][:,-1]

energyPrediction = pd.DataFrame(energyForecast)
energyPrediction['energy_sum'] = train['energy_sum'].values[-80:]

# 6. Obtain performance measures
# Mean Forecast Error (Forecast Bias)
forecast_errors = [energyPrediction['energy_sum'][i]-energyPrediction['energy_sumPred'][i] 
for i in range(len(energyPrediction['energy_sum']))]
bias = sum(forecast_errors)*1.0/len(energyPrediction['energy_sum'])
print('Mean Forecast Error (Forecast Bias): %s' % bias)
# Mean Absolute Error
mae = mean_absolute_error(energyPrediction['energy_sum'], energyPrediction['energy_sumPred'])
print('Mean Absolute Error (MAE): %f' % mae)
# Root Mean Squared Error (RMSE)
mse = mean_squared_error(energyPrediction['energy_sum'], energyPrediction['energy_sumPred'])
rmse = sqrt(mse)
print('Root Mean Squared Error (RMSE): %f' % rmse)

# 7. Visualizing the VAR results
# # Plot Energy Prediction with Confidence Interval
fig = energyPrediction.plot(template='plotly_white',title="ACORN-Group K (MAC000055)",
	labels=dict(value="Energy Consumption (KWh)", variable="Legend"))
# # Plot all features
# fig = forecast.plot(template='plotly_white',title="ACORN-Group K (MAC000055)",
# 	labels=dict(value="Value", variable="Legend"))
fig.show()
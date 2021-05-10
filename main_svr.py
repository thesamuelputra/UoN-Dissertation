# 1. Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set_theme()
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn import metrics

# 2. Importing the dataset
dataset = pd.read_csv("dataset.csv")
dataset.drop(["sunsetTime", "sunriseTime","energy_median","energy_mean",
	"energy_max","energy_count","stdorToU","Acorn_grouped",
	"energy_std","energy_min","precipType","Acorn","windBearing",
	"cloudCover","windSpeed","pressure","visibility","moonPhase",
	"holiday","temperatureMax","temperatureMin"], axis = 1, inplace = True)
mask = dataset['LCLid'].isin(['MAC000002','MAC000606','MAC000096','MAC000050',
	'MAC005555','MAC000098','MAC000225','MAC000059',
	'MAC001201','MAC000584','MAC000055','MAC000101',
	'MAC000114','MAC000107','MAC000363','MAC000025',
	'MAC000006','MAC001600'])
dataset = dataset.loc[mask]
data = dataset.loc[dataset['LCLid'] == "MAC000055"]
data.pop("LCLid")
data = data.reset_index()
X = data.iloc[:,0:1].values.astype(float) # index column
y = data.iloc[:,5:6].values.astype(float) # energy_sum Column

# 3. Feature Scaling
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

# 4. Fitting the Support Vector Regression Models to the dataset
linearRegressor = SVR(kernel='linear')
linearRegressor.fit(X,y)
polyRegressor = SVR(kernel='poly')
polyRegressor.fit(X,y)
rbfRegressor = SVR(kernel='rbf')
rbfRegressor.fit(X,y)

# 5. Predicting a new result
linear_y_pred = sc_y.inverse_transform(
	(linearRegressor.predict(sc_X.transform(np.array([[6.5]]).reshape(1, 1)))))
poly_y_pred = sc_y.inverse_transform(
	(polyRegressor.predict(sc_X.transform(np.array([[6.5]]).reshape(1, 1)))))
rbf_y_pred = sc_y.inverse_transform(
	(rbfRegressor.predict(sc_X.transform(np.array([[6.5]]).reshape(1, 1)))))


print('Linear Kernel Mean Absolute Error (MAE): %f' %
	metrics.mean_absolute_error(y, linearRegressor.predict(X)))
print('Linear Kernel Root Mean Squared Error (RMSE): %f' %
	np.sqrt(metrics.mean_squared_error(y, linearRegressor.predict(X))))
print('Polynomial Kernel Mean Absolute Error (MAE): %f' %
	metrics.mean_absolute_error(y, polyRegressor.predict(X)))
print('Polynomial Kernel Root Mean Squared Error (RMSE): %f' %
	np.sqrt(metrics.mean_squared_error(y, polyRegressor.predict(X))))
print('RBF Kernel Mean Absolute Error (MAE): %f' %
	metrics.mean_absolute_error(y, rbfRegressor.predict(X)))
print('RBF Kernel Root Mean Squared Error (RMSE): %f' %
	np.sqrt(metrics.mean_squared_error(y, rbfRegressor.predict(X))))

# 6. Visualising the SVR results
# Linear Support Vectors
linear_support_X = X[linearRegressor.support_]
linear_support_y = y[linearRegressor.support_]
# Polynomial Support Vectors
poly_support_X = X[polyRegressor.support_]
poly_support_y = y[polyRegressor.support_]
# RBF Support Vectors
rbf_support_X = X[rbfRegressor.support_]
rbf_support_y = y[rbfRegressor.support_]
plt.scatter(X, y, color='dimgray', label='Data', s=15)
# plt.scatter(linear_support_X, linear_support_y, facecolor = 'none', edgecolor= 'salmon', label='Linear Support Vectors', marker='.', s=50)
plt.plot(X, linearRegressor.predict(X), color = 'salmon', label='Linear Model')
# plt.scatter(poly_support_X, poly_support_y, facecolor = 'none', edgecolor= 'skyblue', label='Polynomial Support Vectors', marker='.', s=50)
plt.plot(X, polyRegressor.predict(X), color = 'skyblue', label='Polynomial Model')
# plt.scatter(rbf_support_X, rbf_support_y, facecolor = 'none', edgecolor= 'mediumvioletred', label='RBF Support Vectors', marker='.', s=50)
plt.plot(X, rbfRegressor.predict(X), color = 'mediumvioletred', label='RBF Model')
plt.title('ACORN-Group K (MAC000055)')
plt.xlabel('Time')
plt.ylabel('Energy Consumption (KWh)')
plt.legend()
plt.show()
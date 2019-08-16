import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

dataset = pd.read_csv('Data/RandomForestRegression/Random_Forest_Regression/Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

regressor = RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(X, y)


X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff(Random Forest)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv('profiles.csv')
df['number_of_languages'] = df.speaks.map(lambda languages: len(str(languages).split(',')))

prod_per_year = df.groupby('number_of_languages').income.mean().reset_index()
X = prod_per_year['number_of_languages']
X = X.values.reshape(-1, 1)

y = prod_per_year['income']

plt.scatter(X, y)

start_time = time.time()
regr = linear_model.LinearRegression()
regr.fit(X, y)
elapsed_time = time.time() - start_time
print(elapsed_time)

print(regr.coef_[0])
print(regr.intercept_)

y_predict = regr.predict(X)
plt.plot(X, y_predict)

X_future = np.array(range(5, 10))
X_future = X_future.reshape(-1, 1)
future_predict = regr.predict(X_future)
print(regr.predict([[8]]))
plt.plot(X_future, future_predict)
plt.ylabel('Income')
plt.xlabel('Number of languages')
plt.show()

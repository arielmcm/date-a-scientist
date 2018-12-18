import pandas as pd
import time
from sklearn.neighbors import KNeighborsRegressor

df = pd.read_csv('profiles.csv')
df['number_of_languages'] = df.speaks.map(lambda languages: len(str(languages).split(',')))

prod_per_year = df.groupby('number_of_languages').income.mean().reset_index()
X = prod_per_year['number_of_languages']
X = X.values.reshape(-1, 1)

y = prod_per_year['income']

start_time = time.time()
regressor = KNeighborsRegressor(n_neighbors=1, weights="distance")
regressor.fit(X, y)
elapsed_time = time.time() - start_time
print(elapsed_time)
print(regressor.predict([[8]]))

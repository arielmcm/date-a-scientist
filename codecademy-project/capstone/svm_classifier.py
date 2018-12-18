import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from utils import get_column_mapping
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

df = pd.read_csv('profiles.csv')

df['pets_types'] = get_column_mapping(df.pets)
df['status_level'] = get_column_mapping(df.status)

feature_data = df[['status_level', 'income']]

x = feature_data.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)

feature_data = pd.DataFrame(x_scaled, columns=feature_data.columns)
feature_data['pets_types'] = df.pets_types

feature_data = feature_data.dropna(subset=['pets_types', 'status_level', 'income'])

training_data, validation_data, training_labels, validation_labels = train_test_split(
    feature_data[['status_level', 'income']],
    feature_data['pets_types'],
    test_size=0.2,
    random_state=70
)

start_time = time.time()
classifier = SVC(kernel='linear')
classifier.fit(training_data, training_labels)
score = classifier.score(validation_data, validation_labels)
predictions = classifier.predict(validation_data)
elapsed_time = time.time() - start_time
print(elapsed_time)
print(score)

print(accuracy_score(validation_labels, predictions))
print(recall_score(validation_labels, predictions, average='micro'))
print(precision_score(validation_labels, predictions, average='micro'))
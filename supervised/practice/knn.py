import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# Assign colum names to the dataset
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

# Read dataset to pandas dataframe
dataset = pd.read_csv(url, names=names)

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Scale

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2)

scaler = StandardScaler()
scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

model = KNeighborsClassifier(n_neighbors=5)

model.fit(x_train, y_train)

score = model.score(x_test, y_test)

print(score)

prediction = model.predict(x_test)

print(prediction)
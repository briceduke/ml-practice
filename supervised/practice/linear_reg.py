import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("houses.csv")

area = df["sqft_living"]
price = df['price']

x = np.array(area).reshape(-1,1)
y = np.array(price).reshape(-1,1)

dict = {} # Collect scores and predictions from each iteration

def run(sqft):
    # Test

    area_train, area_test, price_train, price_test = train_test_split(x, y, test_size=0.3)

    model = LinearRegression()

    model.fit(area_train, price_train)

    score = model.score(area_test, price_test)

    # Predict

    prediction = model.predict(np.array([int(sqft)]).reshape(-1, 1)) 

    dict[prediction[0][0]] = score

for _ in range(100):
    run(3000) # Enter area to predict price

result = max(dict, key=dict.get) # Get the prediction with the highest score

print(f"Predicted Price: ${round(result, 2)}")
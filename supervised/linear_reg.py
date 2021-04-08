import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd
import pandas_datareader as web
import operator

from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split

dict = {}

# company = 'BNGO'
# start = dt.datetime(2018,9,20)

company = 'GEVO'
start = dt.datetime(2019,1,1)

# company = 'CCIV'
# start = dt.datetime(2020,9,18)

# company = 'PLTR'
# start = dt.datetime(2020,9,30)

# company = 'SPY'
# start = dt.datetime(2013,1,1)

# company = 'TSLA'
# start = dt.datetime(2019,11,1)

# company = 'DOGE-USD'
# start = dt.datetime(2021,1,1)

# company = 'GME'
# start = dt.datetime(2021,1,25)

# company = 'AMC'
# start = dt.datetime(2021,1,25)

def run(company, start):

    forecast_out = int(1) # predicting days into future

    end = dt.datetime.now()
    data = web.DataReader(company, 'yahoo', start, end)

    df = pd.DataFrame(data, columns=['Close'])
    df = df[['Close']]

    df['Prediction'] = df[['Close']].shift(-forecast_out)

    X = np.array(df.drop(['Prediction'], 1))
    X = preprocessing.scale(X)
    X_forecast = X[-forecast_out:]
    X = X[:-forecast_out]
    y = np.array(df['Prediction'])
    y = y[:-forecast_out]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    # Training
    clf = LinearRegression()
    clf.fit(X_train,y_train)
    # Testing
    confidence = clf.score(X_test, y_test)
    prediction = clf.predict(X_forecast)
    # print(prediction)
    prediction = str(prediction)
    prediction = prediction.replace('[', '')
    prediction = prediction.replace(']', '')
    prediction = float(prediction)
    dict[prediction] = confidence

for i in range(80):
    run(company, start)

ans = max(dict, key=dict.get)
print(f"1 Day Price Prediction: ${round(ans, 2)}")
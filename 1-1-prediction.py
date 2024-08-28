import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Download and prepare the data
data_root = "https://github.com/ageron/data/raw/main/"
lifesat = pd.read_csv(data_root + "lifesat/lifesat.csv")
X = lifesat[["GDP per capita (USD)"]].values
y = lifesat[["Life satisfaction"]].values

# Visualize the data
lifesat.plot(kind='scatter', grid=True,
             x="GDP per capita (USD)", y="Life satisfaction")
plt.axis([5_500, 62_500, 4, 9])
plt.show()

# Select a linear model
model = LinearRegression()

# Train the model
model.fit(X, y)

# Make a prediction for Bangladesh
X_new1 = [[5_599]]  # Bangladesh's GDP per capita in 2020
print(model.predict(X_new1)) # outputs [[4.12860002]]

# Make a prediction for India
X_new2 = [[6_172]]  # India's GDP per capita in 2020
print(model.predict(X_new2)) # outputs [[4.16744312]]
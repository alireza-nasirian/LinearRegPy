import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('TkAgg')

# read data
data = pd.read_csv("data.csv", header=None)

# shuffle data
data.sample(1)

# split to train and test set
training_size = int(0.95 * len(data))
training_data = data.iloc[:training_size, :]
test_data = data.iloc[training_size:, :]

# converting 1d array to 2d array
x = np.vstack([training_data[0], np.ones(len(training_data[0]))]).T

# Linear Regression
m, c = np.linalg.lstsq(x, training_data[1], rcond=None)[0]

# plot train data, test data and fitted line
_ = plt.plot(training_data[0], training_data[1], 'o', label='Original training_data', markersize=10)
_ = plt.plot(test_data[0], test_data[1], 'o', label='Original test_data', markersize=10)
_ = plt.plot(training_data[0], m * training_data[0] + c, 'r', label='Fitted line')
_ = plt.legend()
print("read value")
print(test_data)
print("estimated value")
print(m*test_data[1] + c)
print("error")
print((m*test_data[1] + c) - (test_data[1]))
plt.show()

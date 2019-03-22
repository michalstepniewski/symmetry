import numpy as np
mu, sigma = 0, 0.1 # mean and standard deviation
mu, sigma = 0, 2 # mean and standard deviation
s = np.random.normal(mu, sigma, 1000)
s
s = np.random.normal(mu, sigma, 1)
s
s = np.random.normal(mu, sigma, 1)
s
no_slices = 404
n = 0
while n<=no_slices:
    n += 5 + np.random.normal(mu, sigma, 1)[0]
    print(n)
while n<=no_slices:
    print(range(no_slices[n]))
    n += 5 + np.random.normal(mu, sigma, 1)[0]
while n<=no_slices:
    print(range(no_slices[n]))
    n += 5 + (np.random.normal(mu, sigma, 1))[0]
np.random.normal(mu, sigma, 1)[0]
np.random.normal(mu, sigma, 1).round()[0]
5 + np.random.normal(mu, sigma, 1).round()[0]
int(5 + np.random.normal(mu, sigma, 1).round()[0])
readline.write_history_file('gaussing.py')

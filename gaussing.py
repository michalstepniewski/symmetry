import numpy as np
mu, sigma = 0, 2 # mean and standard deviation
s = np.random.normal(mu, sigma, 1)

no_slices = 404
n = 0
while n<=(no_slices-1):
    print(n)
    n += 5 + int(5 + np.random.normal(mu, sigma, 1).round()[0])


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


import numpy as np
from scipy.stats import binom
import matplotlib.pyplot as plt

k = 10
toursize = np.arange(1,24,1)
print(toursize)
popsize = 800
probabilities = toursize/popsize
print(probabilities)
test = binom.pmf(k=k, n=popsize, p=probabilities)          #P(X  = k)
test1 = binom.cdf(k=k, n=popsize, p=probabilities)         #P(X <= k)
test2 = 1 - binom.cdf(k=k-1, n=popsize, p=probabilities)   #P(X >= k)                       

print(np.round(test, decimals=2))
print(np.round(test1, decimals=2))
print(np.round(test2, decimals=2))

plt.bar(x=toursize, height=test2, width=0.3)
plt.show()
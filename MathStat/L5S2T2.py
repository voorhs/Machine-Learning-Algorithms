X = [69,80,92,81,70,79,78,66,57,77]
Y = [60,84,87,79,73,71,72,67,59,70]
W = 0
for x, y in zip(X, Y):
    W += (x < y)
print(W)
from scipy.stats import wilcoxon
print(wilcoxon(X, Y))
from scipy.stats import binom
print(binom.ppf(1-0.05, len(X), 0.5))
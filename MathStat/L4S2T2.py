X = [
    986, 1005, 991, 994, 983, 1002, 996, 998,1002, 983
]

from scipy.stats import tstd

s = tstd(X)
Xbar = sum(X) / len(X)
Z = (Xbar - 1000) / s * len(X) ** 0.5

from scipy.stats import t

pvalue = 2 * min(t.cdf(Z, len(X) - 1), 1 - t.cdf(Z, len(X) - 1))
print(pvalue)
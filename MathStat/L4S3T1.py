n = 25
s2 = 13.5
sigma0 = 9

Zs = s2 * (n - 1) / sigma0

from scipy.stats import chi2
pvalue = 1 - chi2.cdf(Zs, n - 1)

print(pvalue)
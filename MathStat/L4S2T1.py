alpha0 = 0.05
betta0 = 0.1
a0 = 5
a1 = 5.5
d = 2

from scipy.stats import norm
u095 = norm.ppf(0.95)
u01 = norm.ppf(0.1)
n = d*d*(u095 - u01)*(u095 - u01)/((a1 - a0)*(a1 - a0))
print(n)
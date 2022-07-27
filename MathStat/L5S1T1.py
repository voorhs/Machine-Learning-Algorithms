m = 16
n = 9
sx = 0.91 * m / (m - 1)
sy = 1.51 * n / (n - 1)
K = (sx / m + sy / n) ** 2 / ((sx / m) ** 2 / (m - 1) + (sy / n) ** 2 / (n - 1))
from scipy.stats import t
print(t.ppf(0.3, K // 1))

Xbar = 12.57
Ybar = 11.87

Zs = (Xbar - Ybar) / (sx / m + sy / n) ** 0.5
pr = t.cdf(Zs, K // 1)
pvalue = 2 * min(pr, 1 - pr)
print(pvalue)
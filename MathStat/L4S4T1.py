alpha = 0.02
n = 747
p0 = 0.25
from scipy.stats import norm
u = norm.ppf(alpha)
c1 = n * p0 + (n * p0 * (1 - p0)) ** 0.5 * u
Zs = 60
print(0, c1, "H0" if Zs > c1 else "H1")
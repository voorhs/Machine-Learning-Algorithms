# \alpha = 0.05
# Y = {1\over\sigma^2}\sum_{i=1}^n X_i^2\sim\chi^2(n)
# P(x1 < Y(X,\sigma^2) < x2) = 1 - \alpha

X = [
    -0.05, 0.4, -1.32, 0.59, -2.12, 0.86, -0.62, 0.51, 1.44, 0.49, 1.94, -1.73, 0.67, 0.97, -0.15, 0.8, 0.92, 0.66, -0.06, -1.07
]

alpha = 0.05

from scipy.stats import chi2
x1, x2 = chi2.ppf([alpha / 2, 1 - alpha / 2], len(X))

Xsq = 0
for x in X:
    Xsq += x ** 2

print(Xsq / x2, Xsq / x1)

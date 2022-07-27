X = []
Y = []

with open('cholesterol.txt') as f:
    lines = f.readlines()
    for line in lines:        
        x, y = line.split('\t')
        X.append(int(x))
        Y.append(int(y))

Xbar = sum(X) / len(X)
Ybar = sum(Y) / len(Y)

Qbar = Xbar - Ybar
Sq = 0

for x, y in zip(X, Y):
    Sq += (x - y - Qbar) ** 2

Sq /= len(X) - 1

Zs = Qbar / (Sq / len(X))** 0.5 

from scipy.stats import t
pr = t.cdf(Zs, len(X) - 1)
pvalue = 2 * min(pr, 1 - pr)
print(pvalue)
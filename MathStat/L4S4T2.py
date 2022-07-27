X = [
    1380, 1344, 1356, 1291, 1308, 1271, 1371, 1430, 1381, 1457, 1492, 1240, 1256, 1466, 1214, 1448, 1510, 1395, 1507, 1264, 1293, 1251, 1380, 1386, 1411, 1434, 1302, 1529, 1352, 1494, 1348, 1464, 1286, 1345, 1491, 1259, 1541, 1214, 1310, 1286,
]

from scipy.stats import tstd
Zs = (sum(X) / len(X) - 1222) / tstd(X) * len(X) ** 0.5

print(Zs, len(X))

from scipy.stats import t
pr = t.cdf(Zs, len(X) - 1)
pvalue = 2 * min(pr, 1 - pr)
print(pvalue)
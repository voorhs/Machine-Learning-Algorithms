s1 = [282, 226, 188, 327, 344, 304, 414, 224, 335, 270]
s2 = [417,  851,  742, 1217, 1160,  993,  864,  852, 1286,  988]

m = [sum(s1) / len(s1), sum(s2) / len(s2)]

def var(df):
    mean = sum(df) / len(df)
    var = 0
    for x in df:
        var += (x - mean) ** 2
    var /= len(df)
    return var // 0.000001 * 0.000001

Xs = 0.4 * m[0] + 0.6 * m[1]

sum1 = 0.4 * var(s1) + 0.6 * var(s2)
sum2 = 0.4 * (Xs - m[0]) ** 2 + 0.6 * (Xs - m[1]) ** 2

D = sum1 + sum2

print(Xs, D)
X = [156, 171, 133, 102, 129, 150, 120, 110, 112, 130]
Y = [73, 81, 103, 88, 131, 106, 107, 111, 122, 108]

X = sorted(X)
Y = sorted(Y)

Z = []
rank_sum = 0
i = 0
j = 0

while (i != len(X) and j != len(Y)):
    if (X[i] < Y[j]):
        Z.append((X[i], 'x', i + j + 1))
        i += 1
    else:
        rank_sum += i + j + 1
        Z.append((Y[j], 'y', i + j + 1))
        j += 1

while (i != len(X)):
    Z.append((X[i], 'x', i + j + 1))
    i += 1

while (j != len(Y)):
    rank_sum += i + j + 1
    Z.append((Y[j], 'y', i + j + 1))
    j += 1
            
print(rank_sum)
for t in Z:
    print(t)

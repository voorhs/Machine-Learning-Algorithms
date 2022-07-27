df = []
with open('sample_2_4.txt') as f:
    for line in f.readlines():
        k, r = line.split()
        k = int(k)
        r = (r == 'F')
        df.append((k, r))

# X = (X_1, ..., X_n) and Y = (Y_1, ..., Y_n) are conjugated samples
# probabilities:
#       P(X_i = k, Y_i = F) = (1-p)^{k-1} * p * (1-g)^{k-1}
#       P(X_i = k, Y_i = N) = (1-p)^k * (1-g)^{k-1} * g
# general form of likelihood function:
#       L(X, p, g) = p^a * (1-p)^b * g^c * (1-g)^d

a = 0
b = 0
c = 0
d = 0

for k, r in df:
    a += r
    b += k - r
    c += not r
    d += k - 1

print('p =', a / (a + b))
print('g =', c / (c + d))
import numpy as np
import scipy.stats as stats
import scipy.special as special


def forpirson(x, n, k):
    z0 = np.floor(min(x))
    zk = np.ceil(max(x))
    h = (zk - z0) / k
    w = np.zeros(k)
    i = 0
    for _x in x:
        z0 -= i * h
        i = 0
        while z0 < zk:
            if z0 <= _x <= z0 + h:
                w[i] += 1
            i += 1
            z0 += h
    z0 -= i * h
    w /= n
    z = []
    for i in range(k):
        z.append((z0 + h * i + z0 + h * (i + 1)) / 2)
    z = np.array(z)
    xavg = (z * w).sum()
    xvar = (pow((z - xavg), 2) * w).sum() * k / (k - 1)
    p = []
    for i in range(k):
        p.append((special.erf((z0 + h * (i + 1) - xavg) / np.sqrt(2 * xvar)) - special.erf((z0 + h * (i) - xavg) / np.sqrt(2 * xvar))) / 2)
    p = np.array(p)
    return w, p


def pirson(p1, p2, n):
    if min(p2) > 0:
        return n * (pow((p1 - p2), 2) / p2).sum()
    else:
        return np.inf


def forkolmogorov(x, xj):
    sum = 0
    for _x in x:
        if _x <= xj:
            sum += 1
    return sum


def kolmogorov(x, n, eps = 0.0001):
    D = 0
    for _x in x:
        F = special.erf((_x - np.average(x))/ np.sqrt(2 * np.var(x))) / 2 + 0.5
        Fn__ = forkolmogorov(x, _x + eps) / n
        Fn_ = forkolmogorov(x, _x - eps) / n
        D = max(abs(F - Fn_), abs(F - Fn__), D)
    return np.sqrt(n) * D


def kolmogorovsmirnov(x1, x2, n, eps = 0.0001):
    D = 0
    for _x in x1:
        F11_ = forkolmogorov(x1, _x - eps) / n
        F11__ = forkolmogorov(x1, _x + eps) / n
        F21 = forkolmogorov(x2, _x) / n
        for __x in x2:
            F12 = forkolmogorov(x1, __x) / n
            F22_ = forkolmogorov(x2, __x - eps) / n
            F22__ = forkolmogorov(x2, __x + eps) / n
            D = max(abs(F11_ - F21), abs(F11__ - F21), abs(F12 - F22_), abs(F12 - F22__), D)
    return np.sqrt(n / 2) * D


if __name__ == '__main__':
    n = 1000
    alpha = 0.05
    k = int(np.floor(np.log2(n)) + 1)
    x = np.random.normal(3, np.sqrt(5), n)
    w, p = forpirson(x, n, k)
    ans = pirson(w, p, n)
    chi2 = stats.chi2.ppf(1 - alpha, k - 1)
    print('1)Критерий Пирсона')
    print(ans, chi2, ans < chi2, 'Без шума')

    y = x + 0.5 * np.random.standard_cauchy(n)
    w, p = forpirson(y, n, k)
    ans = pirson(w, p, n)
    print(ans, chi2, ans < chi2, 'Для смещённой')

    z = x + 0.3 * np.random.uniform(-1, 1, n)
    w, p = forpirson(z, n, k)
    ans = pirson(w, p, n)
    print(ans, chi2, ans < chi2, 'Для смещённой с другим шумом')

    ans = kolmogorov(x, n)
    kolm = stats.kstwobign.ppf(1 - alpha)
    print('2) Критерий Колмогорова')
    print(ans, kolm, ans < kolm, 'Без шума')

    ans = kolmogorov(y, n)
    print(ans, kolm, ans < kolm, 'Для смещённой')

    ans = kolmogorov(z, n)
    print(ans, kolm, ans < kolm, 'Для смещённой с другим шумом')

    ans = kolmogorovsmirnov(x[:500], x[500:], 500)
    print('3)Критерий Колмогорова-Смирнова для двух частей одной выборки')
    print(ans, kolm, ans < kolm)

    x2 = np.random.normal(3, np.sqrt(5.5), n)
    ans = kolmogorovsmirnov(x[:500], x2[:500], 500)
    print(ans, kolm, ans < kolm)





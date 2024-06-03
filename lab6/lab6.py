import numpy as np
import scipy.stats as stats
import scipy.special as special


def w(x, y):
    d = {"+": 0, "-": 0}
    n = len(x)
    for i in range(n):
        if x[i] < y[i]:
            d["+"] += 1
        else:
            d["-"] += 1
    return min(d["+"], d["-"])


def t20(w, n):
    s = 0
    for i in range(w + 1):
        s += special.comb(n, i, exact=True)
    return s / pow(2, n)


def t1000(w, n):
    return (w - n / 2) / np.sqrt(n / 4)


def cor(x):
    s = 0
    n = len(x)
    summ = sum(x)
    for i in range(n - 1):
        s += x[i] * x[i + 1]
    r = (n * s - summ ** 2 + n * x[0] * x[n - 1]) / (n * sum(x ** 2) - summ ** 2)
    m = -1 / (n - 1)
    d = n * (n - 3) / (n + 1) / (n - 1) ** 2
    return (r - m) / np.sqrt(d)


def h(x):
    n = len(x)
    med = np.median(x)
    s = 0
    for i in range(n):
        s += i * (x[i] - med) ** 2
    H = s / (n - 1) / sum((x - med) ** 2)
    m = 1 / 2
    d = (n + 1) / 6 / (n - 1) / (n + 2)
    return (H - m) / np.sqrt(d)


if __name__ == '__main__':
    n = 1000
    alpha = 0.05

    print("Предполагается, что t∈{α/2, 1-α/} => H0 - {x-y=0} ")
    x11 = np.random.normal(3, np.sqrt(5), n)
    y11 = np.random.normal(3, np.sqrt(6), n)
    ans1 = t20(w(x11[:20], y11[:20]), 20)
    ans2 = t1000(w(x11, y11), n)
    N01 = stats.norm.ppf(1 - alpha / 2)
    print("Для N(3,5), N(3,6)")
    print("n = 20 : ", ans1, "{", alpha / 2, ";", 1 - alpha / 2, "}", alpha / 2 < ans1 < 1 - alpha / 2)
    print("n = 1000 : ", ans2, "{", -N01, ";", N01, "}", -N01 < ans2 < N01)

    x12 = x11
    y12 = np.random.normal(3.5, np.sqrt(6), n)
    ans1 = t20(w(x12[:20], y12[:20]), 20)
    ans2 = t1000(w(x12, y12), n)
    print("Для N(3,5), N(3.5,6)")
    print("n = 20 : ", ans1, "{", alpha / 2, ";", 1 - alpha / 2, "}", alpha / 2 < ans1 < 1 - alpha / 2)
    print("n = 1000 : ", ans2, "{", -N01, ";", N01, "}", -N01 < ans2 < N01)

    x21 = x11
    ans = cor(x21)
    print("")
    print("Предполагаем, что коэффициент автокорреляции = 0")
    print("N(3,5) : ", ans, "{", -N01, ";", N01, "}", -N01 < ans < N01)

    y21 = []
    y21.append(x21[0] - 2 * x21[len(x21) - 1])
    for i in range(1, len(x21)):
        y21.append(x21[i] + 2 * x21[i - 1])
    y21 = np.array(y21)
    ans = cor(y21)
    print("y[j+1] = x[j] + 2x[j-1]; "
          "y[1] = x[1] - 2x[n]: ", ans, "{", -N01, ";", N01, "}", -N01 < ans < N01)


    x31 = x11
    ans = h(x31)
    print("")
    print("Предполагаем, что дисперсии x[j] одинаковые")
    print("N(3,5) : ", ans, "{", -N01, ";", N01, "}", -N01 < ans < N01)

    y31 = list(x11[:500]) + list(y11[:500])
    y31 = np.array(y31)
    ans = h(y31)
    print("Для j∈[1;500]: N(3,5), Для j∈[501;1000]: N(3,6) : ", ans, "{", -N01, ";", N01, "}", -N01 < ans < N01)
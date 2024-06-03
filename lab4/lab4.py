import numpy as np
import scipy.stats as stats


def cor(x, y, n):
    avrgx = np.average(x)
    avrgy = np.average(y)
    sx = 0
    sy = 0
    sxy = 0
    for i in range(n):
        sx += pow(x[i] - avrgx, 2)
        sy += pow(y[i] - avrgy, 2)
        sxy += (x[i] - avrgx) * (y[i] - avrgy)
    return sxy / np.sqrt(sx * sy)


def t_estimation1(r, n):
    return r * np.sqrt(n - 1) / np.sqrt(1 - r ** 2)


def t_estimation2(x, y, n):
    return (np.average(x) - np.average(y)) / np.sqrt((2 / n) * (n - 1) * (np.var(x) * n / (n - 1) + np.var(y) * n /(n -1)) / (2 * n - 2))


def t_estimation3(z, n):
    return np.average(z) * np.sqrt(n) / np.sqrt(np.var(z))


def f_estimation(x, y):
    if np.var(x) < np.var(y):
        return np.var(x) / np.var(y)
    else:
        return np.var(y) / np.var(x)


def z_estimation1(p, p1, p2, n):
    return (p1 - p2 - 1 / (4 * n)) / np.sqrt(p * (1 - p) * (2 / n))


def z_estimation2(x1, x2, n):
    return (np.average(x1) - np.average(x2)) / np.sqrt((np.average(x1) + np.average(x2)) / n)


if __name__ == '__main__':

    alpha = 0.05
    n = 1000
    n2 = 500

    normal_data = np.random.normal(3, np.sqrt(5), n)
    e = np.random.normal(0, 1, n2)
    x = normal_data[:n2]
    x1 = normal_data[n2:]
    t_est1 = t_estimation1(cor(x, x1, n2), n2)
    print("T РєСЂРёС‚РµСЂРёР№ РєРѕСЂСЂРµР»СЏС†РёРё 1", -stats.t.ppf(1 - alpha / 2, n2 - 2), stats.t.ppf(1 - alpha / 2, n2 - 2), t_est1)

    y1 = 5 * x1 - 7 + 0.3 * e
    t_est1 = t_estimation1(cor(x1, y1, n2), n2)
    print("T РєСЂРёС‚РµСЂРёР№ РєРѕСЂСЂРµР»СЏС†РёРё 2", -stats.t.ppf(1 - alpha / 2, n2 - 2), stats.t.ppf(1 - alpha / 2, n2 - 2), t_est1)

    t_est2 = t_estimation2(x, x1, n2)
    print("T РєСЂРёС‚РµСЂРёР№ РґР»СЏ РЅРµСЃРІСЏР·РЅС‹С… РІС‹Р±РѕСЂРѕРє 1", -stats.t.ppf(1 - alpha / 2, 2 * n2 - 2), stats.t.ppf(1 - alpha / 2, 2 * n2 - 2), t_est2)

    y2 = np.random.normal(2, np.sqrt(6), n2)
    t_est2 = t_estimation2(x, y2, n2)
    print("T РєСЂРёС‚РµСЂРёР№ РґР»СЏ РЅРµСЃРІСЏР·РЅС‹С… РІС‹Р±РѕСЂРѕРє 2", -stats.t.ppf(1 - alpha / 2, 2 * n2 - 2), stats.t.ppf(1 - alpha / 2, 2 * n2 - 2), t_est2)

    y3 = 5 * x
    t_est3 = t_estimation3(x - y3, n2)
    print("T РєСЂРёС‚РµСЂРёР№ РґР»СЏ СЃРІСЏР·РЅС‹С… РІС‹Р±РѕСЂРѕРє 1", -stats.t.ppf(1 - alpha / 2, n2 - 1), stats.t.ppf(1 - alpha / 2, n2 - 1), t_est3)

    y3 = 5 * x + 0.4 + 0.3 * e
    t_est3 = t_estimation3(x - y3, n2)
    print("T РєСЂРёС‚РµСЂРёР№ РґР»СЏ СЃРІСЏР·РЅС‹С… РІС‹Р±РѕСЂРѕРє 2", -stats.t.ppf(1 - alpha / 2, n2 - 1), stats.t.ppf(1 - alpha / 2, n2 - 1), t_est3)

    f_est = f_estimation(x, x1)
    print("Р¤ РєСЂРёС‚РµСЂРёР№ Р¤РёС€РµСЂР°1", stats.f.ppf(alpha / 2, n2 - 1, n2 - 1), stats.f.ppf(1 - alpha / 2, n2 - 1, n2 - 1), f_est)

    f_est = f_estimation(x, y2)
    print("Р¤ РєСЂРёС‚РµСЂРёР№ Р¤РёС€РµСЂР°2", stats.f.ppf(alpha / 2, n2 - 1, n2 - 1), stats.f.ppf(1 - alpha / 2, n2 - 1, n2 - 1), f_est)

    b1 = np.random.binomial(n2, 0.3)
    b2 = np.random.binomial(n2, 0.3)
    b3 = np.random.binomial(n, 0.4)
    p = (b1 + b2) / n
    p1 = b1 / n2
    p2 = b2 / n2
    z_est1 = z_estimation1(p, p1, p2, n2)
    print("Z РєСЂРёС‚РµСЂРёР№ (Р‘РµСЂРЅСѓР»Р»Рё)1", stats.norm.ppf(alpha / 2), stats.norm.ppf(1 - alpha / 2), z_est1)

    p = (b1 + b3) / n
    p1 = b1 / n2
    p2 = b3 / n2
    z_est1 = z_estimation1(p, p1, p2, n2)
    print("Z РєСЂРёС‚РµСЂРёР№ (Р‘РµСЂРЅСѓР»Р»Рё)2", stats.norm.ppf(alpha / 2), stats.norm.ppf(1 - alpha / 2), z_est1)

    p = np.random.poisson(3, n)
    p1 = p[:n2]
    p2 = p[n2:]
    p3 = np.random.poisson(4, n2)
    z_est2 = z_estimation2(p1, p2, n2)
    print("Z РєСЂРёС‚РµСЂРёР№ (РџСѓР°СЃСЃРѕРЅ)1", stats.norm.ppf(alpha / 2), stats.norm.ppf(1 - alpha / 2), z_est2)

    z_est2 = z_estimation2(p1, p3, n2)
    print("Z РєСЂРёС‚РµСЂРёР№ (РџСѓР°СЃСЃРѕРЅ)2", stats.norm.ppf(alpha / 2), stats.norm.ppf(1 - alpha / 2), z_est2)

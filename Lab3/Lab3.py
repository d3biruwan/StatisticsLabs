import numpy as np
import matplotlib.pyplot as plt


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


def fish(r):
    return 1 / 2 * np.log((1 + r) / (1 - r))


def zhuk(r, n):
    return r / np.sqrt(1 - pow(r, 2)) * np.sqrt(n - 2)


def linereg(xp, x, y, r):
    return r * (xp - np.average(x)) * np.sqrt(np.var(y)) / np.sqrt(np.var(x)) + np.average(y)


if __name__ == '__main__':

    n = 500
    n2 = 250

    normal_data = np.random.normal(3, np.sqrt(5), n)
    e = np.random.normal(0, 1, n2)
    x = normal_data[:n2]
    y = normal_data[n2:]
    r = cor(x, y, n2)
    z = fish(r)
    t = zhuk(r, n2)
    print(r, "- Оценка коэффициента корреляции (Независимы)")
    print(z, "- Преобразование Фишера")
    print(t, "- Преобразование Жуковского")

    y1 = 5 * x - 7 + 0.3 * e
    r1 = cor(x, y1, n2)
    z1 = fish(r1)
    t1 = zhuk(r1, n2)
    print(r1, "- Оценка коэффициента корреляции (Зависимы)")
    print(z1, "- Преобразование Фишера")
    print(t1, "- Преобразование Жуковского")

    plt.subplot(1, 2, 1)
    plt.scatter(x, y1, c="blue")

    x2 = np.linspace(-4, 10, 1000)
    y2 = linereg(x2, x, y1, r1)
    plt.title("Линейная регрессия")
    plt.plot(x2, y2, c="black")
    print("Линейная регрессия")
    print(r1 * np.sqrt(np.var(y1)) / np.sqrt(np.var(x)), r1 * np.sqrt(np.var(y1)) / np.sqrt(np.var(x)) * (-np.average(x)) + np.average(y1))

    y3 = 4 * np.sin(x) + 0.3 * e
    s = np.zeros(8)
    for i in range(n2):
        s[0] += x[i]
        s[1] += pow(x[i], 2)
        s[2] += pow(x[i], 3)
        s[3] += pow(x[i], 4)
        s[4] += pow(x[i], 5)
        s[5] += pow(x[i], 6)
        s[6] += pow(x[i], 7)
        s[7] += pow(x[i], 8)

    a = [[n2, s[0], s[1], s[2], s[3]],
         [s[0], s[1], s[2], s[3], s[4]],
         [s[1], s[2], s[3], s[4], s[5]],
         [s[2], s[3], s[4], s[5], s[6]],
         [s[3], s[4], s[5], s[6], s[7]]]
    b = [sum(y3), sum(y3 * x), sum(y3 * pow(x, 2)), sum(y3 * pow(x, 3)), sum(y3 * pow(x, 4))]
    x4 = np.linspace(-4, 10, 1000)
    betha = np.linalg.solve(np.array(a), np.array(b))
    y4 = betha[0] + betha[1] * x4 + betha[2] * pow(x4, 2) + betha[3] * pow(x4, 3) + betha[4] * pow(x4, 4)
    plt.subplot(1, 2, 2)
    plt.title("Многомерная регрессия")
    plt.scatter(x, y3, c="red")
    plt.plot(x4, y4, c="black")
    print(betha)

    plt.show()






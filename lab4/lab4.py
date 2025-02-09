import math
from scipy.stats import ttest_ind
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import t
from scipy import stats
import numpy as np
from scipy import stats
def T_krit(x, y):
    """
    Функция для проверки значимости коэффициента корреляции.
    Аргументы:
    x, y -- входные массивы данных
    Возвращает:
    Вывод результатов проверки значимости коэффициента корреляции.
    """
    rho = np.corrcoef(x, y)[0, 1]  # Вычисляем коэффициент корреляции
    alpha = 0.05  # Уровень значимости
    t_stat = rho * np.sqrt((len(x) - 2) / (1 - rho**2))  # Вычисляем значение t-статистики
    t_crit = stats.t.ppf(1 - alpha / 2, len(x) - 2)  # Вычисляем критическое значение t-статистики


    print(f"Значение статистики t: {t_stat:.4f}")
    print(f"Критическое значение t: ±{t_crit:.4f}")


    # Проверка уровня значимости
    if -t_crit < t_stat < t_crit:
        print("Принимаем нулевую гипотезу")
    else:
        print("Отвергаем нулевую гипотезу")
    print("=" * 50)

n = 1000
sample = np.random.normal(3, math.sqrt(5), n)
x, y = sample[:500], sample[500:]

print("Проверка значимости коэффициента корреляции")
print()
print("Выборка из N(3,5), разделенная пополам:")

T_krit(x, y)

x= np.random.normal(3, math.sqrt(5), n)
epsilon=np.random.normal(0, 1, n)
y=5*x-7+0.3*epsilon


print('Выборка y=5x-7+0.3e' )
T_krit(x,y)

def T_krit_mean(x, y):
    """
    Функция для проверки значимости разницы средних двух выборок.

    Аргументы:
    x, y -- входные массивы данных

    Возвращает:
    Вывод результатов проверки значимости разницы средних.
    """
    n = len(x)  # Предполагаем, что длины выборок равны
    mean1 = np.mean(x)  # Вычисляем среднее первой выборки
    mean2 = np.mean(y)  # Вычисляем среднее второй выборки
    std1 = np.std(x)  # Вычисляем стандартное отклонение первой выборки
    std2 = np.std(y)  # Вычисляем стандартное отклонение второй выборки
    alpha = 0.05  # Уровень значимости
    S = np.sqrt(((n - 1) * std1 ** 2 + (n - 1) * std2 ** 2) / (2 * n - 2))  # Вычисляем объединенное стандартное отклонение

    # Вычисление статистики t и p-value
    t_stat = (mean1 - mean2) / (S * np.sqrt((1 / n) + (1 / n)))
    df = 2 * n - 2  # Число степеней свободы
    t_crit = stats.t.ppf(1 - alpha / 2, df)  # Вычисляем критическое значение t-статистики

    print(f"Значение статистики t: {t_stat:.4f}")
    print(f"Критическое значение t: ±{t_crit:.4f}")

    # Проверка уровня значимости
    if -t_crit < t_stat < t_crit:
        print("Принимаем нулевую гипотезу")
    else:
        print("Отвергаем нулевую гипотезу")
    print("=" * 50)

n=1000
x= np.random.normal(3, math.sqrt(5), n)
y= np.random.normal(2, math.sqrt(6), n)
x1,x2=x[:500], x[500:]

print()
print("Проверка значимости разницы средних двух выборок")
print()
print('Выборка разделенная пополам:')
T_krit_mean(x1,x2)

print('Две разные выборки:')
T_krit_mean(x,y)

def diff(z):
    """
    Функция для проверки значимости среднего значения выборки.

    Аргументы:
    z -- входной массив данных

    Возвращает:
    Вывод результатов проверки значимости среднего значения.
    """
    mean_z = np.mean(z)  # Вычисляем среднее значение выборки
    std_z = np.std(z, ddof=1)  # Вычисляем стандартное отклонение выборки (с поправкой на смещение)
    alpha = 0.05  # Уровень значимости

    # Вычисление статистики t и p-value
    t_stat = mean_z / (std_z / np.sqrt(len(z)))
    df = len(z) - 1  # Число степеней свободы
    t_crit = stats.t.ppf(1 - alpha / 2, df)  # Вычисляем критическое значение t-статистики


    print(f"Значение статистики t: {t_stat:.4f}")
    print(f"Критическое значение t: ±{t_crit:.4f}")


    # Проверка уровня значимости
    if -t_crit < t_stat < t_crit:
        print("Принимаем нулевую гипотезу")
    else:
        print("Отвергаем нулевую гипотезу")
    print("=" * 50)

n=100
x= np.random.normal(3, math.sqrt(5), n)
y= 5*x
z=x-y

print()
print("Проверка значимости среднего значения выборки")
print()
print('z=x-5x')
diff(z)

x= np.random.normal(3, math.sqrt(5), n)
epsilon=np.random.normal(0, 1, n)
y=5*x+0.4+0.3*epsilon

print('z=x-5x+0.4+0.3e')
diff(x-y)

def Fisher(x, y):
    """
    Функция для проверки равенства дисперсий двух выборок с помощью F-теста.

    Аргументы:
    x, y -- входные массивы данных

    Возвращает:
    Вывод результатов проверки равенства дисперсий.
    """
    alpha = 0.05  # Уровень значимости
    std1 = np.std(x)  # Вычисляем стандартное отклонение первой выборки
    std2 = np.std(y)  # Вычисляем стандартное отклонение второй выборки
    f = (std1 ** 2) / (std2 ** 2)  # Вычисляем значение F-статистики
    F_l = stats.f.ppf(alpha / 2, len(x) - 1, len(y) - 1)  # Вычисляем левое критическое значение F-статистики
    F_r = stats.f.ppf(1 - alpha / 2, len(x) - 1, len(y) - 1)  # Вычисляем правое критическое значение F-статистики


    print(f"Значение статистики F: {f:.4f}")
    print(f"Критические значения F: {F_l:.4f}, {F_r:.4f}")


    # Проверка уровня значимости
    if F_l < f < F_r:
        print("Принимаем нулевую гипотезу")
    else:
        print("Отвергаем нулевую гипотезу")
    print("=" * 50)

n=1000
x= np.random.normal(3, math.sqrt(5), n)
y= np.random.normal(2, math.sqrt(6), n)

x1,x2=x[:500], x[500:]

print()
print("Проверка равенства дисперсий двух выборок (F-тест)")
print()
print('Выборка разделенная пополам')
Fisher(x1,x2)

print('Две разные выборки')
Fisher(x,y)

def binom(x, y, n):
    """
    Функция для проверки значимости разницы долей двух выборок.

    Аргументы:
    x -- количество успехов в первой выборке
    y -- количество успехов во второй выборке
    n -- общее число наблюдений в каждой выборке

    Возвращает:
    Вывод результатов проверки значимости разницы долей.
    """
    l = x  # Количество успехов в первой выборке
    n1 = n  # Общее число наблюдений в первой выборке
    m = y  # Количество успехов во второй выборке
    n2 = n  # Общее число наблюдений во второй выборке
    alpha = 0.05  # Уровень значимости
    p = (l + m) / (n1 + n2)  # Оценка общей доли успехов
    Z = (l / n1 - m / n2 - 1 / (2 * (n1 + n2))) / (np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2)))  # Вычисление Z-статистики


    print(f"Значение статистики Z: {Z:.4f}")
    z_crit = stats.norm.ppf(1 - alpha / 2)  # Вычисление критического значения Z-статистики
    print(f"Критическое значение Z: ±{z_crit:.4f}")


    # Проверка уровня значимости
    if -z_crit < Z < z_crit:
        print("Принимаем нулевую гипотезу")
    else:
        print("Отвергаем нулевую гипотезу")
    print("=" * 50)

n=1000
x= np.random.binomial(n,0.3)
y= np.random.binomial(n,0.4)
x1= np.random.binomial( n,0.3)

print()
print("Проверка значимости разницы долей двух выборок")
print()
print('Две разные выборки:')
binom(x,y,n)

print('Выборка разделенная пополам:')
binom(x,x1,n)

def puasson(x, y, n):
    """
    Функция для проверки значимости разницы средних двух выборок с распределением Пуассона.

    Аргументы:
    x -- первая выборка с распределением Пуассона
    y -- вторая выборка с распределением Пуассона
    n -- размер каждой выборки

    Возвращает:
    Вывод результатов проверки значимости разницы средних.
    """
    alpha = 0.05  # Уровень значимости
    Z = (np.mean(x) - np.mean(y)) / np.sqrt(np.mean(x) / n + np.mean(y) / n)  # Вычисление Z-статистики


    print(f"Значение статистики Z: {Z:.4f}")
    z_crit = stats.norm.ppf(1 - alpha / 2)  # Вычисление критического значения Z-статистики
    print(f"Критическое значение Z: ±{z_crit:.4f}")


    # Проверка уровня значимости
    if -z_crit < Z < z_crit:
        print("Принимаем нулевую гипотезу")
    else:
        print("Отвергаем нулевую гипотезу")
    print("=" * 50)

n=1000
x= np.random.poisson(3,n)
y= np.random.poisson(4,n)
x1= np.random.poisson(3,n)

print()
print("Проверка значимости разницы средних двух выборок с распределением Пуассона")
print()
print('Две разные выборки:')
puasson(x,y,n)

print('Выборка разделенная пополам:')
puasson(x,x1,n)




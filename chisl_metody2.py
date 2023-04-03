import numpy as np
import matplotlib.pyplot as plt
import math


def fun(x):
    return math.sin(x) / (1 + x ** 2)


def lagrange_interpolation(x, y):
    """
    Реализация интерполяционной формулы Лагранжа для заданных точек (x, y)
    """
    n = len(x)
    poly = np.poly1d(0)  # создаем пустой многочлен f(x) = 0

    for i in range(n):
        poly_i = np.poly1d(y[i])  # f_i(x) = y_i
        for j in range(n):
            if i != j:
                poly_i *= np.poly1d([1, -x[j]])  # умножаем на (x - xj) f_i(x) = y_i * (x - x_j)
                poly_i /= x[i] - x[
                    j]  # делим на (xi - xj)   f_i(x) = y_i * (x - x_0) / (x_1 - x_0) * (x - x_2) / (x_1 - x_2)
        poly += poly_i

    # Форматируем многочлен в строку
    return poly


# Считываем входные данные из файла input.txt
x, y = [], []
with open("input.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        x_i, y_i = line.split()
        x.append(float(x_i))
        y.append(float(y_i))

# Вычисляем многочлен
polynom = lagrange_interpolation(x, y)
x_p = [(x[i // 4] + x[(i + 1) // 4] + x[(i + 2) // 4] + x[(i + 3) // 4]) / 4 for i in range(4 * len(x) - 3)]

steps = 1000
h = (x[-1] - x[0]) / steps
x_or = [x[0] + h * i for i in range(steps + 1)]

with open("output.txt", "w") as f:
    f.writelines([f"{x}\t{y}\n" for x, y in zip(x_p, [polynom(i) for i in x_p])])



plt.scatter(x, y, color='red')  # то что было

plt.plot(x_p, [polynom(i) for i in x_p], color='blue') # Лагранж

plt.plot(x_or, [fun(x) for x in x_or], color='green') # по чему строили

plt.show()

# Записываем результат в файл output.txt



# f(x) = a_6 * x ^ 6 + a_5 * x ^ 5 + ... a_1 * x + a_0

# F(x) = x ^ 2

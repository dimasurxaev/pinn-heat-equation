#! python3.7
# -*- coding: utf-8 -*-

import matplotlib
matplotlib.use('TkAgg')

from numpy import zeros, linspace, exp, linalg, real
from matplotlib.pyplot import style, figure, axes, show
from celluloid import Camera


def u_init(x):
    return -x + 1.0


def u_left(t):
    return exp(-t)


# y = (u1, u2, ..., uN, u0)
def f(y, t, h):
    f_val = zeros(N + 1)

    # Уравнение для u1: слева стоит граничное значение u0
    f_val[0] = exp(y[0] ** 2) - y[0] * (y[0] - y[N]) / h

    # Уравнения для u2, ..., uN
    for n in range(1, N):
        f_val[n] = exp(y[n] ** 2) - y[n] * (y[n] - y[n - 1]) / h

    # Алгебраическое уравнение: u0 = u_left(t)
    f_val[N] = y[N] - u_left(t)

    return f_val


def f_y(y, t, h):
    jac = zeros((N + 1, N + 1))

    # Производные для первого уравнения (u1)
    jac[0, 0] = (-2.0 * y[0] + y[N]) / h + 2.0 * y[0] * exp(y[0] ** 2)
    jac[0, N] = y[0] / h

    # Производные для уравнений u2, ..., uN
    for n in range(1, N):
        jac[n, n] = (-2.0 * y[n] + y[n - 1]) / h + 2.0 * y[n] * exp(y[n] ** 2)
        jac[n, n - 1] = y[n] / h

    # Производная алгебраического уравнения
    jac[N, N] = 1.0

    return jac


def D_matrix():
    d = zeros((N + 1, N + 1))
    for i in range(N):
        d[i, i] = 1.0
    return d


a = 0.0
b = 1.0
t_0 = 0.0
T = 0.3

alpha = (1.0 + 1.0j) / 2.0   # CROS1
# alpha = 1.0                # DIRK1

N = 200
M = 300

h = (b - a) / N
x = linspace(a, b, N + 1)

tau = (T - t_0) / M
t = linspace(t_0, T, M + 1)

# Y[m] = (u1, u2, ..., uN, u0)
Y = zeros((M + 1, N + 1))

# Начальное условие
for n in range(N):
    Y[0, n] = u_init(x[n + 1])

Y[0, N] = u_left(t_0)

D = D_matrix()

for m in range(M):
    J = f_y(Y[m], t[m], h)
    rhs = f(Y[m], t[m] + tau / 2.0, h)

    w_1 = linalg.solve(D - alpha * tau * J, rhs)
    Y[m + 1] = Y[m] + tau * real(w_1)

    # Поддерживаем алгебраическое ограничение
    Y[m + 1, N] = u_left(t[m + 1])

# Сборка для графика: (u0, u1, ..., uN)
U = zeros((M + 1, N + 1))
U[:, 0] = Y[:, N]
U[:, 1:] = Y[:, :N]

style.use('dark_background')
fig = figure()
camera = Camera(fig)
ax = axes(xlim=(a, b), ylim=(0.0, 3.0))
ax.set_xlabel('x')
ax.set_ylabel('u')

for m in range(M + 1):
    ax.plot(x, U[m], color='y', ls='-', lw=2)
    camera.snap()

animation = camera.animate(interval=20, repeat=False, blit=False)

show()
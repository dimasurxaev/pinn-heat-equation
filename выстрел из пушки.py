from numpy import *
from matplotlib.pyplot import *


def f(u, g, k, mass):
    f = zeros(4)
    f[0] = u[2]
    f[1] = u[3]
    f[2] = 0.0 - k / mass * sqrt(u[2] ** 2 + u[3] ** 2) * u[2]
    f[3] = -g - k / mass * sqrt(u[2] ** 2 + u[3] ** 2) * u[3]
    return f


t_0 = 0.0
T = 7.0
x_0 = 0.0
y_0 = 0.0
v_0 = 150.0
alpha = pi / 4
g = 9.81
k = 10.0
mass = 500.0

M = 50


s = 4  # ERK4
b = zeros(s)
a = zeros((s, s))
c = zeros(s)
b[0] = 1 / 6
b[1] = 1 / 3
b[2] = 1 / 3
b[3] = 1 / 6
a[1, 0] = 1 / 2
a[2, 0] = 0.0
a[2, 1] = 1 / 2
a[3, 0] = 0.0
a[3, 1] = 0.0
a[3, 2] = 1.0
c[0] = 0.0
c[1] = 1 / 2
c[2] = 1 / 2
c[3] = 1.0


tau = (T - t_0) / M
t = linspace(t_0, T, M + 1)


u = zeros((M + 1, 4))
ur = zeros((M + 1, 4))

u[0, 0] = x_0  # начальный x
u[0, 1] = y_0  # начальный y
u[0, 2] = v_0 * cos(alpha)  # начальная vx
u[0, 3] = v_0 * sin(alpha)  # начальная vy
ur[0, 0] = x_0  # начальный x
ur[0, 1] = y_0  # начальный y
ur[0, 2] = v_0 * cos(alpha)  # начальная vx
ur[0, 3] = v_0 * sin(alpha)  # начальная vy


for m in range(M):
    u[m + 1] = u[m] + tau * f(u[m], g, k, mass)


for m in range(M):
    w = zeros((s, 4))

    for i in range(s):
        adjustment_1 = zeros(4)

        for l in range(i):
            adjustment_1 = adjustment_1 + a[i, l] * w[l]

        w[i] = f(ur[m] + tau * adjustment_1, g, k, mass)

    adjustment_2 = zeros(4)
    for i in range(s):
        adjustment_2 = adjustment_2 + b[i] * w[i]

    ur[m + 1] = ur[m] + tau * adjustment_2


figure()
plot(u[:, 0], u[:, 1], "-ro", markersize=5, label="Метод Эйлера")

plot(ur[:, 0], ur[:, 1], "-go", markersize=5, label="Метод Рунге-Кутты")
legend()

title("Траектория движения тела")
xlabel("x")
ylabel("y")
xlim((0, 1.62 * 80))
ylim((0, 80))
show()

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

alpha = 0.015  # коэффициент теплопроводности


class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1),
        )

    def forward(self, x, t):

        inputs = torch.cat([x, t], dim=1)
        return self.net(inputs)


model = PINN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# (физический лосс)
N_f = 1000
x_f = torch.rand(N_f, 1) * 1.0
T_MAX = 5.0
t_f = torch.rand(N_f, 1) * T_MAX
N_ic = 1000
x_ic = torch.rand(N_ic, 1) * 1.0
t_ic = torch.zeros(N_ic, 1)

# значения температуры в начальный момент
u_ic_true = torch.sin(np.pi * x_ic)

for epoch in range(5000):
    optimizer.zero_grad()

    x_f.requires_grad = True
    t_f.requires_grad = True

    u = model(x_f, t_f)

    u_t = torch.autograd.grad(
        u, t_f, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]

    u_x = torch.autograd.grad(
        u, x_f, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]

    u_xx = torch.autograd.grad(
        u_x, x_f, grad_outputs=torch.ones_like(u_x), create_graph=True
    )[0]

    residual = u_t - alpha * u_xx
    loss_physics = torch.mean(residual**2)

    u_ic_pred = model(x_ic, t_ic)
    loss_ic = torch.mean((u_ic_pred - u_ic_true) ** 2)
    total_loss = loss_physics + loss_ic

    # Обратное распространение
    total_loss.backward()
    optimizer.step()

    if epoch % 1000 == 0:
        print(f"ЦИкл {epoch:4d}, Loss: {loss_physics.item():.4e}")

print("Обучение завершено")

# Тестовые данные: температура в начальный момент времени u(x,0)
x_test = torch.linspace(0, 1, 100).reshape(-1, 1)
t_test = torch.zeros_like(x_test)

model.eval()
with torch.no_grad():
    u_pred = model(x_test, t_test).numpy()

t_moments = [0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]

plt.figure(figsize=(10, 6))
for t_val in t_moments:
    t_test = torch.ones_like(x_test) * t_val
    u_pred = model(x_test, t_test).detach().numpy()
    plt.plot(x_test.numpy(), u_pred, label=f"t = {t_val}")

plt.xlabel("x (координата)")
plt.ylabel("Температура")
plt.title("Распределение температуры в разные моменты времени")
plt.legend()
plt.grid(True)
plt.show()

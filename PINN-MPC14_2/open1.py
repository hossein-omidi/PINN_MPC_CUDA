import numpy as np
import pandas as pd
import torch
from model import get_model
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Parameters
w0 = 0.0082
ws = 0.0080
Cpa = 1000
Cpw = 4180
hw = 800 * 1000
hfg = 2500 * 1000
rho_a = 1.18
rho_w = 1000
Vt = 400
Vc = 1
fa_dot = 2.6
fw_dot = .9 / 1000
T0 = 32
dTc = 6
M_dot0 = 0.000115
Q_dot0 = 10 * 1000

# Derived parameters
alpha1 = 1 / Vt
alpha2 = 1 / (rho_a * Vt)
alpha3 = 1 / Vc
beta1 = hfg / (Cpa * Vt)
beta2 = (rho_w * Cpw * dTc) / (rho_a * Cpa * Vc)
gamma1 = 1 / (rho_a * Cpa * Vt)
gamma2 = hw / (Cpa * Vc)


def nonlinear_system(t, x, u):
    Tt, wt, Ts = x
    fa_dot, fw_dot, u3 = u

    M_dot0_unc = M_dot0 * (1 + 0.025 * np.random.uniform(-1, 1))
    Q_dot0_unc = Q_dot0 * (1 + 0.025 * np.random.uniform(-1, 1))

    dTt = (1 / (rho_a * Cpa * Vt)) * (Q_dot0_unc - hfg * M_dot0_unc) + \
          ((fa_dot * hfg) / (Cpa * Vt)) * (wt - ws) - (fa_dot / Vt) * (Tt - Ts)
    dwt = (M_dot0_unc / (rho_a * Vt)) - (fa_dot / Vt) * (wt - ws) + u3
    dTs = (fa_dot / Vc) * (Tt - Ts) + \
          ((1 - 0.75) * fa_dot / Vc) * (T0 - Tt) - \
          (fa_dot * hw / (Cpa * Vc)) * ((1 - 0.75) * w0 + 0.75 * wt - ws) - \
          (fw_dot * rho_w * Cpw * dTc) / (rho_a * Cpa * Vc)

    return np.array([dTt, dwt, dTs])


def rk4_step(f, t, x, u, dt):
    k1 = f(t, x, u)
    k2 = f(t + dt / 2, x + dt * k1 / 2, u)
    k3 = f(t + dt / 2, x + dt * k2 / 2, u)
    k4 = f(t + dt, x + dt * k3, u)
    return x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


if __name__ == "__main__":
    dt_rk4 = 0.01
    simulation_time = 500.0
    n_steps = int(simulation_time / dt_rk4)
    initial_state = np.array([26, 0.008, 18])
    control_input = np.array([fa_dot, fw_dot, 0.0])

    model = get_model("lstm", {'input_dim': 6, 'hidden_dim': 128, 'layer_dim': 10, 'output_dim': 3})
    model.load_state_dict(torch.load("PINN_STZ.pth", map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    states = np.zeros((n_steps, 3), dtype=np.float32)
    pinn_predictions = np.zeros((n_steps, 3), dtype=np.float32)
    states[0] = initial_state

    time = np.arange(n_steps) * dt_rk4

    for t in range(1, n_steps):
        states[t] = rk4_step(nonlinear_system, time[t - 1], states[t - 1], control_input, dt_rk4)
        input_tensor = torch.FloatTensor(np.hstack((states[t - 1], control_input))).to(device).unsqueeze(0)
        pinn_predictions[t] = model(input_tensor).cpu().detach().numpy().squeeze()

    plt.figure(figsize=(12, 6))
    plt.plot(time, states[:, 0], label='RK4 Tt')
    plt.plot(time, pinn_predictions[:, 0], label='PINN Tt', linestyle='--')
    plt.plot(time, states[:, 1], label='RK4 wt')
    plt.plot(time, pinn_predictions[:, 1], label='PINN wt', linestyle='--')
    plt.plot(time, states[:, 2], label='RK4 Ts')
    plt.plot(time, pinn_predictions[:, 2], label='PINN Ts', linestyle='--')
    plt.xlabel('Time (s)')
    plt.ylabel('States')
    plt.title('Comparison of RK4 and PINN Model Simulation')
    plt.legend()
    plt.grid()
    plt.show()

    differences = states - pinn_predictions
    plt.figure(figsize=(12, 6))
    plt.plot(time, differences[:, 0], label='Tt Difference')
    plt.plot(time, differences[:, 1], label='wt Difference')
    plt.plot(time, differences[:, 2], label='Ts Difference')
    plt.xlabel('Time (s)')
    plt.ylabel('Differences')
    plt.title('Differences between RK4 and PINN Model States')
    plt.legend()
    plt.grid()
    plt.show()

import numpy as np
import pandas as pd
import torch
from model import get_model
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Parameters (unchanged)
w0 = 8.2
ws = 8
Cpa = 1000
Cpw = 4180
hw = 800 * 1000
hfg = 2500 * 1000
rho_a = 1.18
rho_w = 1000
Vt = 400
Vc = 1
fa_dot = 2.6
fw_dot = 0.9 / 1000
T0 = 32
dTc = 6
M_dot0 = 0.000115
Q_dot0 = 20 * 1000

# Derived parameters (unchanged)
alpha1 = 1 / Vt
alpha2 = 1 / (rho_a * Vt)
alpha3 = 1 / Vc
beta1 = hfg / (Cpa * Vt)
beta2 = (rho_w * Cpw * dTc) / (rho_a * Cpa * Vc)
gamma1 = 1 / (rho_a * Cpa * Vt)
gamma2 = hw / (Cpa * Vc)

# Nonlinear system function (unchanged)
def nonlinear_system(t, x, u):
    Tt, wt, Ts = x
    fa_dot, fw_dot, u3 = u

    dTt = (1 / (rho_a * Cpa * Vt)) * (Q_dot0 - hfg * M_dot0) + \
          ((fa_dot * hfg) / (1000 * Cpa * Vt)) * (wt - ws) - (fa_dot / Vt) * (Tt - Ts)

    dwt = 1000 * (M_dot0 / (rho_a * Vt)) - (fa_dot / Vt) * (wt - ws) + u3

    dTs = (fa_dot / Vc) * (Tt - Ts) + \
          (0.25 * fa_dot / Vc) * (T0 - Tt) - \
          (fa_dot * hw / (1000 * Cpa * Vc)) * (0.25 * w0 + 0.75 * wt - ws) - \
          (fw_dot * rho_w * Cpw * dTc) / (rho_a * Cpa * Vc)

    return np.array([dTt, dwt, dTs])

# RK4 step function (unchanged)
def rk4_step(f, t, x, u, dt):
    k1 = f(t, x, u)
    k2 = f(t + dt / 2, x + dt * k1 / 2, u)
    k3 = f(t + dt / 2, x + dt * k2 / 2, u)
    k4 = f(t + dt, x + dt * k3, u)
    return x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

# Open-loop simulation (unchanged)
def run_open_loop_simulation():
    dt_rk4 = 0.1
    simulation_time = 1500.0
    n_steps = int(simulation_time / dt_rk4)
    initial_state = np.array([26, 8, 18])
    states = np.zeros((n_steps, 3), dtype=np.float32)
    states[0] = initial_state
    time = np.arange(n_steps) * dt_rk4

    # Control inputs as a sine wave for fa_dot
    control_inputs = np.zeros((n_steps, 3))
    for t in range(n_steps):
        # Sine wave for fa_dot: amplitude 0.5, frequency 0.01 Hz, centered around 2.6
        fa_dot_t = 2.6 + 0.5 * np.sin(2 * np.pi * 0.01 * time[t])
        control_inputs[t] = [fa_dot_t, fw_dot, 0.001]

    # Simulation loop with time-varying control inputs
    for t in range(1, n_steps):
        states[t] = rk4_step(nonlinear_system, t * dt_rk4, states[t - 1], control_inputs[t], dt_rk4)

    return pd.DataFrame({
        'Time': time,
        'Tt': states[:, 0],
        'wt': states[:, 1],
        'Ts': states[:, 2]
    })

# Updated compare_with_PINN function with sine wave control inputs
def compare_with_PINN(results):
    model = get_model("lstm", {
        'input_dim': 6, 'hidden_dim': 256,
        'layer_dim': 12, 'output_dim': 3
    })
    model.load_state_dict(torch.load("PINN_STZ_colab.pth", map_location=torch.device('cpu'), weights_only=True))
    model.to(device)
    model.eval()

    # Prepare data for PINN model
    inputs = results[['Tt', 'wt', 'Ts']].values[:-1]
    time = results['Time'].values[:-1]

    # Generate sine wave control inputs matching the simulation
    control_inputs = np.zeros((len(inputs), 3))
    for t in range(len(inputs)):
        fa_dot_t = 2.6 + 0.5 * np.sin(2 * np.pi * 0.01 * time[t])  # Same sine wave as in simulation
        control_inputs[t] = [fa_dot_t, fw_dot, 0.0]  # fw_dot and u3 remain constant

    # Model Test Plot
    x_test = torch.FloatTensor(np.hstack((inputs, control_inputs))).to(device)
    yt = model(x_test).cpu().detach().numpy()

    return yt

if __name__ == "__main__":
    results = run_open_loop_simulation()
    pinn_results = compare_with_PINN(results)

    time = results['Time'][:len(pinn_results)]

    # Plotting (unchanged)
    plt.figure(figsize=(12, 6))
    plt.plot(time, results['Tt'][:len(pinn_results)], label='RK4 Tt')
    plt.plot(time, pinn_results[:, 0], label='PINN Tt', linestyle='--')
    plt.xlabel('Time (s)')
    plt.ylabel('Tt State')
    plt.title('Comparison of RK4 and PINN Model for Tt State with Sine Wave Input')
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(time, results['wt'][:len(pinn_results)], label='RK4 wt')
    plt.plot(time, pinn_results[:, 1], label='PINN wt', linestyle='--')
    plt.xlabel('Time (s)')
    plt.ylabel('wt State')
    plt.title('Comparison of RK4 and PINN Model for wt State with Sine Wave Input')
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(time, results['Ts'][:len(pinn_results)], label='RK4 Ts')
    plt.plot(time, pinn_results[:, 2], label='PINN Ts', linestyle='--')
    plt.xlabel('Time (s)')
    plt.ylabel('Ts State')
    plt.title('Comparison of RK4 and PINN Model for Ts State with Sine Wave Input')
    plt.legend()
    plt.grid()
    plt.show()

    # Differences (unchanged)
    differences_Tt = results['Tt'][:len(pinn_results)] - pinn_results[:, 0]
    plt.figure(figsize=(12, 6))
    plt.plot(time, differences_Tt, label='Tt Difference')
    plt.xlabel('Time (s)')
    plt.ylabel('Tt Difference')
    plt.title('Difference between RK4 and PINN Model for Tt State with Sine Wave Input')
    plt.legend()
    plt.grid()
    plt.show()

    differences_wt = results['wt'][:len(pinn_results)] - pinn_results[:, 1]
    plt.figure(figsize=(12, 6))
    plt.plot(time, differences_wt, label='wt Difference')
    plt.xlabel('Time (s)')
    plt.ylabel('wt Difference')
    plt.title('Difference between RK4 and PINN Model for wt State with Sine Wave Input')
    plt.legend()
    plt.grid()
    plt.show()

    differences_Ts = results['Ts'][:len(pinn_results)] - pinn_results[:, 2]
    plt.figure(figsize=(12, 6))
    plt.plot(time, differences_Ts, label='Ts Difference')
    plt.xlabel('Time (s)')
    plt.ylabel('Ts Difference')
    plt.title('Difference between RK4 and PINN Model for Ts State with Sine Wave Input')
    plt.legend()
    plt.grid()
    plt.show()
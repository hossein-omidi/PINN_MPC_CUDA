import numpy as np
import pandas as pd
import torch
from model import get_model
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Parameters
w0 = 8.2        # g water / kg dry air
ws = 8.0        # g water / kg dry air
Cpa = 1000      # J/(kg·K)
Cpw = 4180      # J/(kg·K)
hw = 800 * 1000 # J/kg
hfg = 2500 * 1000 # J/kg
rho_a = 1.18    # kg/m³
rho_w = 1000    # kg/m³
Vt = 400        # m³
Vc = 1          # m³
fa_dot = 2.6    # m³/s
fw_dot = 0.9    # g/s
T0 = 32         # °C
dTc = 6         # °C
M_dot0 = 0.000115  # kg/s (corrected)
Q_dot0 = 15 * 1000 # J/s

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

    # Introduce uncertainties
    M_dot0_unc = M_dot0 * (1 + 0.025 * np.random.uniform(-1, 1))
    Q_dot0_unc = Q_dot0 * (1 + 0.025 * np.random.uniform(-1, 1))

    dTt = (1 / (rho_a * Cpa * Vt)) * (Q_dot0 - hfg * M_dot0) + ((fa_dot * hfg) / (1000 * Cpa * Vt)) * (wt - ws) - (
            fa_dot / Vt) * (Tt - Ts)
    dwt = 1000 * (M_dot0 / (rho_a * Vt)) - ((fa_dot / Vt) / 1000) * (wt - ws) + u3
    dTs = (fa_dot / Vc) * (Tt - Ts) + ((1 - 0.75) * fa_dot / Vc) * (T0 - Tt) - (fa_dot * hw / (1000 * Cpa * Vc)) * (
            (1 - 0.75) * w0 + 0.75 * wt - ws) - (fw_dot * rho_w * Cpw * dTc) / (1000 * rho_a * Cpa * Vc)

    return np.array([dTt, dwt, dTs])


def rk4_step(f, t, x, u, dt):
    k1 = f(t, x, u)
    k2 = f(t + dt / 2, x + dt * k1 / 2, u)
    k3 = f(t + dt / 2, x + dt * k2 / 2, u)
    k4 = f(t + dt, x + dt * k3, u)
    return x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def run_open_loop_simulation():
    # Timing parameters
    dt_rk4 = 0.5  # Simulation time step (100 Hz)
    simulation_time = 3000.0  # Total simulation time in seconds

    # Number of steps
    n_steps = int(simulation_time / dt_rk4)

    # Initial state near operating point
    initial_state = np.array([26, 8, 18])

    # Control inputs (constant for open loop)
    control_input = np.array([fa_dot, fw_dot, 0])

    # Initialize states
    states = np.zeros((n_steps, 3), dtype=np.float32)
    states[0] = initial_state

    # Open-loop simulation loop
    for t in range(1, n_steps):
        states[t] = rk4_step(nonlinear_system, t * dt_rk4, states[t - 1], control_input, dt_rk4)

    # Time array for plotting
    time = np.arange(n_steps) * dt_rk4

    return pd.DataFrame({
        'Time': time,
        'Tt': states[:, 0],
        'wt': states[:, 1],
        'Ts': states[:, 2]
    })


def compare_with_PINN(results):
    # Initialize model
    model = get_model("lstm", {
        'input_dim': 6, 'hidden_dim': 256,
        'layer_dim': 8, 'output_dim': 3
    })

    # Safe model loading
    model.load_state_dict(torch.load("PINN_STZ.pth",
                                     map_location=torch.device('cpu'),
                                     weights_only=True))  # Security fix
    model.to(device)
    model.eval()

    # Prepare data for PINN model
    inputs = results[['Tt', 'wt', 'Ts']].values[:-1]
    control_input = np.array([fa_dot, fw_dot, 0.0])
    control_inputs = np.repeat([control_input], len(inputs), axis=0)

    # Model Test Plot
    x_test = torch.FloatTensor(np.hstack((inputs, control_inputs))).to(device)  # Shape: (N-1, 6)
    yt = model(x_test).cpu().detach().numpy()  # Convert to numpy

    return yt


if __name__ == "__main__":
    results = run_open_loop_simulation()

    # Compare with PINN model
    pinn_results = compare_with_PINN(results)

    # Plot the open-loop simulation results and PINN results
    time = results['Time'][:len(pinn_results)]

    # Plot Tt state comparison
    plt.figure(figsize=(12, 6))
    plt.plot(time, results['Tt'][:len(pinn_results)], label='RK4 Tt')
    plt.plot(time, pinn_results[:, 0], label='PINN Tt', linestyle='--')
    plt.xlabel('Time (s)')
    plt.ylabel('Tt State')
    plt.title('Comparison of RK4 and PINN Model for Tt State')
    plt.legend()
    plt.grid()
    plt.show()

    # Plot wt state comparison
    plt.figure(figsize=(12, 6))
    plt.plot(time, results['wt'][:len(pinn_results)], label='RK4 wt')
    plt.plot(time, pinn_results[:, 1], label='PINN wt', linestyle='--')
    plt.xlabel('Time (s)')
    plt.ylabel('wt State')
    plt.title('Comparison of RK4 and PINN Model for wt State')
    plt.legend()
    plt.grid()
    plt.show()

    # Plot Ts state comparison
    plt.figure(figsize=(12, 6))
    plt.plot(time, results['Ts'][:len(pinn_results)], label='RK4 Ts')
    plt.plot(time, pinn_results[:, 2], label='PINN Ts', linestyle='--')
    plt.xlabel('Time (s)')
    plt.ylabel('Ts State')
    plt.title('Comparison of RK4 and PINN Model for Ts State')
    plt.legend()
    plt.grid()
    plt.show()

    # Calculate and plot the differences for Tt state
    differences_Tt = results['Tt'][:len(pinn_results)] - pinn_results[:, 0]
    plt.figure(figsize=(12, 6))
    plt.plot(time, differences_Tt, label='Tt Difference')
    plt.xlabel('Time (s)')
    plt.ylabel('Tt Difference')
    plt.title('Difference between RK4 and PINN Model for Tt State')
    plt.legend()
    plt.grid()
    plt.show()

    # Calculate and plot the differences for wt state
    differences_wt = results['wt'][:len(pinn_results)] - pinn_results[:, 1]
    plt.figure(figsize=(12, 6))
    plt.plot(time, differences_wt, label='wt Difference')
    plt.xlabel('Time (s)')
    plt.ylabel('wt Difference')
    plt.title('Difference between RK4 and PINN Model for wt State')
    plt.legend()
    plt.grid()
    plt.show()

    # Calculate and plot the differences for Ts state
    differences_Ts = results['Ts'][:len(pinn_results)] - pinn_results[:, 2]
    plt.figure(figsize=(12, 6))
    plt.plot(time, differences_Ts, label='Ts Difference')
    plt.xlabel('Time (s)')
    plt.ylabel('Ts Difference')
    plt.title('Difference between RK4 and PINN Model for Ts State')
    plt.legend()
    plt.grid()
    plt.show()

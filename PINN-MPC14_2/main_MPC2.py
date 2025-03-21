import numpy as np
import pandas as pd
import torch
from model import get_model
from MPC import cost_fun_mimo
from plotting import MPCplot
import matplotlib

matplotlib.use('TkAgg')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# System parameters
w0 = 8.2  # kg/kg, initial humidity ratio
ws = 8.0  # kg/kg, supply air humidity ratio
Cpa = 1000  # J/(kg·K)
Cpw = 4180  # J/(kg·K)
hw = 800000  # J/kg
hfg = 2500000  # J/kg
rho_a = 1.18  # kg/m³
rho_w = 1000  # kg/m³
Vt = 400  # m³
Vc = 1  # m³
T0 = 32  # °C
dTc = 6  # K
M_dot0 = 0.000115  # kg/s
Q_dot0 = 20000  # W


def generate_setpoint_mpc(t_now):
    """Dynamic reference generator for all three states"""
    if t_now < 500:
        Tt_ref = 23 + 1.5 * np.sin(0.05 * t_now)
        wt_ref = 8 + 1 * np.sin(0.025 * t_now)
        Ts_ref = 18
    else:
        step_index = int((t_now - 500) // 200)
        Tt_ref = 23 + (step_index % 4) * 0.5
        wt_ref = 8 + (step_index % 4) * .5
        Ts_ref = 18
    return np.array([Tt_ref, wt_ref, Ts_ref])


def system_dynamics(t, x, u):
    """True system dynamics for RK4 simulation"""
    Tt, wt, Ts = x
    fa_dot, fw_dot, u3 = u

    # Differential equations
    dTt = (1 / (rho_a * Cpa * Vt)) * (Q_dot0 - hfg * M_dot0) + \
          ((fa_dot * hfg) / (1000 * Cpa * Vt)) * (wt - ws) - (fa_dot / Vt) * (Tt - Ts)

    dwt = 1000 * (M_dot0 / (rho_a * Vt)) - (fa_dot / Vt) * (wt - ws) + u3

    dTs = (fa_dot / Vc) * (Tt - Ts) + \
          (0.25 * fa_dot / Vc) * (T0 - Tt) - \
          (fa_dot * hw / (1000 * Cpa * Vc)) * (0.25 * w0 + 0.75 * wt - ws) - \
          (fw_dot * rho_w * Cpw * dTc) / (rho_a * Cpa * Vc)

    return np.array([dTt, dwt, dTs])


def rk4_step(x, u, dt):
    """4th-order Runge-Kutta integrator"""
    k1 = system_dynamics(None, x, u)
    k2 = system_dynamics(None, x + dt / 2 * k1, u)
    k3 = system_dynamics(None, x + dt / 2 * k2, u)
    k4 = system_dynamics(None, x + dt * k3, u)
    return x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def run_mpc_simulation():
    # Timing parameters
    dt_rk4 = 0.1  # 100Hz simulation
    dt_mpc = 0.2  # 5Hz control updates
    total_time = 1500  # 10 minutes simulation
    n_steps = int(total_time / dt_rk4)
    mpc_interval = int(dt_mpc / dt_rk4)

    # Initialize system
    # Model initialization with checks
    model = get_model("lstm", {
        'input_dim': 6, 'hidden_dim': 256,
        'layer_dim': 12, 'output_dim': 3
    })
    model.load_state_dict(torch.load("PINN_STZ_colab.pth", map_location=device))
    model.eval()

    model.eval()

    # Configure bounds and weights
    bounds = {
        'control': {
            'fa_dot': (0.1, 5.0),
            'fw_dot': (0.0, 0.003),
            'u3': (-1e-2, 1e-2)
        },
        'state': {
            'Tt': (16.0, 32.0),
            'wt': (6.5, 9.5),
            'Ts': (10.0, 26.0)
        }
    }

    # Weight matrices with enhanced tracking emphasis
    W = np.diag([100, 100, 1])  # 100x higher weight for Tt/wt tracking
    R = np.diag([1, 1, 5])  # Reduced control effort penalty (softer constraints)

    # Initialize states and controls
    current_state = np.array([23.0, 8.0, 18.0])  # Initial condition
    states = np.zeros((n_steps, 3))
    controls = np.zeros((n_steps, 3))
    states[0] = current_state
    current_control = np.array([2.0, 0.001, 0.0])

    # Main simulation loop
    next_mpc_step = mpc_interval
    for t in range(1, n_steps):
        current_time = t * dt_rk4
        current_ref = generate_setpoint_mpc(current_time)

        # MPC control update
        if t >= next_mpc_step:
            try:
                current_control, pred, _ = cost_fun_mimo(
                    current_states=states[t - 1],
                    prev_controls=current_control,
                    references=current_ref,
                    bounds=bounds,
                    model=model,
                    W=W,
                    R=R,  # Added control effort weighting matrix
                    lambda_tracking=1,
                    lambda_terminal=0.1,
                    lambda_integral=.51,
                    w_state_con=.1,
                    w_control_con=.1,
                    s=1e-4,
                    max_iter=10,
                    dt=dt_mpc
                )
                next_mpc_step += mpc_interval
            except Exception as e:
                print(f"MPC failure at {current_time:.1f}s: {str(e)}")

        # Apply control and simulate
        controls[t] = current_control
        states[t] = rk4_step(states[t - 1], controls[t], dt_rk4)

        # Monitoring
        if t % 100 == 0:
            print(f"Time: {current_time:.1f}s | "
                  f"Tt: {states[t, 0]:.2f}C (ref: {current_ref[0]:.2f}) | "
                  f"wt: {states[t, 1]:.5f} (ref: {current_ref[1]:.5f}) | "
                  f"Ts: {states[t, 2]:.2f}C (ref: {current_ref[2]:.2f})")

    # Compile results
    time_axis = np.arange(n_steps) * dt_rk4
    return pd.DataFrame({
        'Time': time_axis,
        'Tt_actual': states[:, 0],
        'wt_actual': states[:, 1],
        'Ts_actual': states[:, 2],
        'fa_dot': controls[:, 0],
        'fw_dot': controls[:, 1],
        'u3': controls[:, 2],
        'Tt_ref': [generate_setpoint_mpc(t)[0] for t in time_axis],
        'wt_ref': [generate_setpoint_mpc(t)[1] for t in time_axis],
        'Ts_ref': [generate_setpoint_mpc(t)[2] for t in time_axis]
    })


if __name__ == "__main__":
    results = run_mpc_simulation()
    MPCplot(results[['Tt_actual', 'wt_actual', 'Ts_actual',
                     'fa_dot', 'fw_dot', 'u3',
                     'Tt_ref', 'wt_ref', 'Ts_ref']])
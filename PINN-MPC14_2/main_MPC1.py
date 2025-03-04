import numpy as np
import pandas as pd
import torch
from model import get_model
from MPC import MPCConfig, cost_function
from plotting import BldgEnergyPlot, testplot,MPCplot
import matplotlib

matplotlib.use('TkAgg')



def run_mpc():
    # Load references and initial conditions
    states_df = pd.read_csv('mpc_states.csv')
    references = states_df[['Reference Tt', 'Reference wt']].values
    initial_states = states_df[['MPC Tt', 'MPC wt', 'MPC Ts']].values[0]

    # Load trained model
    model = get_model("ann", {
        'input_dim': 6, 'hidden_dim': 64,
        'layer_dim': 3, 'output_dim': 3
    })
    model.load_state_dict(torch.load("PINN_STZ.pth"))
    model.eval()

    # MPC configuration
    config = MPCConfig()
    n_steps = len(references)
    controls = np.zeros((n_steps, 3))
    states = np.zeros((n_steps, 3))
    states[0] = initial_states

    # Main control loop
    for t in range(1, n_steps):
        try:
            # Get current state and reference
            current_state = states[t - 1]
            current_ref = references[t - 1]

            # MPC optimization
            u_opt, pred, energy = cost_function(
                current_state,
                controls[t - 1],
                current_ref,
                model,
                config
            )

            # Apply controls and update state
            controls[t] = u_opt
            states[t] = pred

        except Exception as e:
            print(f"Error at step {t}: {str(e)}")
            controls[t] = controls[t - 1]
            states[t] = states[t - 1]

    # Save results for analysis
    results = pd.DataFrame({
        'Time': np.arange(n_steps),
        'Tt_actual': states[:, 0],
        'wt_actual': states[:, 1],
        'Ts_actual': states[:, 2],
        'Tt_ref': references[:, 0],
        'wt_ref': references[:, 1],
        'fa_dot': controls[:, 0],
        'fw_dot': controls[:, 1],
        'u3': controls[:, 2]
    })
    results.to_csv('mpc_results.csv', index=False)

    return results

if __name__ == "__main__":
    results = run_mpc()
    MPCplot(results)  # Now accepts single DataFrame argument
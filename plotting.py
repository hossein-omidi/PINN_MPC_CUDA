import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter


def BldgEnergyPlot(t_room, t_amb, t_set, ac_onoff, tevap, power):
    fig, (ax1, ax2) = plt.subplots(2, 1, layout='constrained')
    value = np.timedelta64(1, 'm')
    times = np.arange(np.datetime64('2023-08-15 09:00:00'), np.datetime64('2023-08-15 17:00:00'), value)
    ax1.plot(times, t_room[:-1], label='T_room')
    ax1.plot(times, t_amb, label='T_amb')
    ax1.plot(times, tevap, label='T_supply')
    ax1.plot(times, t_set+2, label='T_set_ub')
    ax1.plot(times, t_set-2, label='T_set_lb')
    ax1.set_xlabel('Time [min]')
    ax1.set_ylabel('Temperature [C]')
    ax1.legend(loc='best',fontsize="8",ncol=2)
    date_form = DateFormatter("%H:%M")
    ax1.xaxis.set_major_formatter(date_form)
    ax2.plot(times, ac_onoff)
    ax2.set_xlabel('Time [min]')
    ax2.set_ylabel('AC on/off')
    ax2.xaxis.set_major_formatter(date_form)
    plt.show()


def testplot(yact, ypred):
    # Adjusting the time axis to match the number of time steps
    times = np.arange(len(yact))  # Generate correct x-axis based on the length of actual values

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 12), layout='constrained')  # 3 subplots

    # Plot Tt (actual vs predicted)
    ax1.plot(times, yact[:, 0], label="Tt_actual", linestyle="-", color='blue')
    ax1.plot(times, ypred[:, 0], label="Tt_predicted", linestyle="--", color='red')
    ax1.set_title('Tt (Actual vs Predicted)')
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Tt [°C]")
    ax1.legend()

    # Plot wt (actual vs predicted)
    ax2.plot(times, yact[:, 1], label="wt_actual", linestyle="-", color='blue')
    ax2.plot(times, ypred[:, 1], label="wt_predicted", linestyle="--", color='red')
    ax2.set_title('wt (Actual vs Predicted)')
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("wt [kg]")
    ax2.legend()

    # Plot Ts (actual vs predicted)
    ax3.plot(times, yact[:, 2], label="Ts_actual", linestyle="-", color='blue')
    ax3.plot(times, ypred[:, 2], label="Ts_predicted", linestyle="--", color='red')
    ax3.set_title('Ts (Actual vs Predicted)')
    ax3.set_xlabel("Time Step")
    ax3.set_ylabel("Ts [°C]")
    ax3.legend()

    plt.tight_layout()
    plt.show()


def MPCplot(results_df):
    """Updated plotting function using DataFrame input"""
    # Create 6 subplots: 3 for states and 3 for controls
    fig, ax = plt.subplots(6, 1, figsize=(10, 15), sharex=True)

    # State Plots
    state_labels = ['Tt [°C]', 'wt [kg/kg]', 'Ts [°C]']
    for i in range(3):
        ax[i].plot(results_df[f'{state_labels[i].split()[0]}_actual'], label='Actual')
        if i < 2:  # Only Tt and wt have references
            ax[i].plot(results_df[f'{state_labels[i].split()[0]}_ref'], 'r--', label='Reference')
        ax[i].set_ylabel(state_labels[i])
        ax[i].legend()

    # Control Plots
    control_labels = ['fa_dot', 'fw_dot', 'u3']
    for i, col in enumerate(control_labels):
        # Use indices 3, 4, 5 for control plots
        ax[i + 3].plot(results_df[col], label=f'{col} [{"kg/s" if i < 2 else "−"}]')
        ax[i + 3].set_ylabel(f'{col} [{"kg/s" if i < 2 else "−"}]')
        ax[i + 3].legend()

    ax[-1].set_xlabel('Time Step')  # Label the x-axis on the last subplot
    plt.tight_layout()
    plt.show()

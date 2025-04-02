clc; clear; close all;

% Load data from files
ann_data = readtable('ann.csv');
rnn_data = readtable('rnn.csv');
lstm_data = readtable('lstm.csv');

% Extract time and shared state variables
Time = ann_data.Time;
Tt = ann_data.Tt;
wt = ann_data.wt;
Ts = ann_data.Ts;

% Extract PINN predictions from each method
PINN_Tt_ANN = ann_data.PINN_Tt;
PINN_wt_ANN = ann_data.PINN_wt;
PINN_Ts_ANN = ann_data.PINN_Ts;

PINN_Tt_RNN = rnn_data.PINN_Tt;
PINN_wt_RNN = rnn_data.PINN_wt;
PINN_Ts_RNN = rnn_data.PINN_Ts;

PINN_Tt_LSTM = lstm_data.PINN_Tt;
PINN_wt_LSTM = lstm_data.PINN_wt;
PINN_Ts_LSTM = lstm_data.PINN_Ts;

% Define line styles
line_styles = {'-', '--', '-.', ':'}; % Solid, Dashed, Dash-dot, Dotted
colors = lines(4);

% Plot results
figure('Units','inches', 'Position', [1, 1, 6.5, 8]); % IEEE standard size

% Subplot for Tt comparison
subplot(3,1,1);
plot(Time, Tt, line_styles{1}, 'Color', colors(1,:), 'LineWidth', 1.5); hold on;
plot(Time, PINN_Tt_ANN, line_styles{2}, 'Color', colors(2,:), 'LineWidth', 1.2);
plot(Time, PINN_Tt_RNN, line_styles{3}, 'Color', colors(3,:), 'LineWidth', 1.2);
plot(Time, PINN_Tt_LSTM, line_styles{4}, 'Color', colors(4,:), 'LineWidth', 1.2);
ylabel('$T_t$ (°C)', 'Interpreter', 'latex', 'FontSize', 12);
title('Comparison of Zone Temperature ($T_t$)', 'Interpreter', 'latex', 'FontSize', 14);
grid on;
legend('RK4 (solid)', 'ANN (dashed)', 'RNN (dash-dot)', 'LSTM (dotted)', 'Location', 'Best', 'FontSize', 10);
hold off;

% Subplot for wt comparison
subplot(3,1,2);
plot(Time, wt, line_styles{1}, 'Color', colors(1,:), 'LineWidth', 1.5); hold on;
plot(Time, PINN_wt_ANN, line_styles{2}, 'Color', colors(2,:), 'LineWidth', 1.2);
plot(Time, PINN_wt_RNN, line_styles{3}, 'Color', colors(3,:), 'LineWidth', 1.2);
plot(Time, PINN_wt_LSTM, line_styles{4}, 'Color', colors(4,:), 'LineWidth', 1.2);
ylabel('$w_t$ (g/kg)', 'Interpreter', 'latex', 'FontSize', 12);
title('Comparison of Zone Humidity ($w_t$)', 'Interpreter', 'latex', 'FontSize', 14);
grid on;
legend('RK4 (solid)', 'ANN (dashed)', 'RNN (dash-dot)', 'LSTM (dotted)', 'Location', 'Best', 'FontSize', 10);
hold off;

% Subplot for Ts comparison
subplot(3,1,3);
plot(Time, Ts, line_styles{1}, 'Color', colors(1,:), 'LineWidth', 1.5); hold on;
plot(Time, PINN_Ts_ANN, line_styles{2}, 'Color', colors(2,:), 'LineWidth', 1.2);
plot(Time, PINN_Ts_RNN, line_styles{3}, 'Color', colors(3,:), 'LineWidth', 1.2);
plot(Time, PINN_Ts_LSTM, line_styles{4}, 'Color', colors(4,:), 'LineWidth', 1.2);
ylabel('$T_s$ (°C)', 'Interpreter', 'latex', 'FontSize', 12);
xlabel('Time (s)', 'Interpreter', 'latex', 'FontSize', 12);
title('Comparison of Supply Air Temperature ($T_s$)', 'Interpreter', 'latex', 'FontSize', 14);
grid on;
legend('RK4 (solid)', 'ANN (dashed)', 'RNN (dash-dot)', 'LSTM (dotted)', 'Location', 'Best', 'FontSize', 10);
hold off;

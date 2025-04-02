% Load the CSV file
data = readtable('mpc_simulation_20250401_140846.csv');

% Extract data
Time = data.Time;
Tt_actual = data.Tt_actual;
wt_actual = data.wt_actual;
Ts_actual = data.Ts_actual;
fa_dot = data.fa_dot;
fw_dot = data.fw_dot;
u3 = data.u3;
u3= u3*400;
Tt_ref = data.Tt_ref;
wt_ref = data.wt_ref;
Ts_ref = data.Ts_ref;

% Define IEEE-friendly font size and line width
fontSize = 11;
lineWidth = 1.5;

%% Plot Control Variables in Subplots
figure;
set(gcf, 'Position', [100, 100, 600, 500]); % Adjust figure size

subplot(3,1,1);
plot(Time, fa_dot, 'r', 'LineWidth', lineWidth);
grid on;
xlabel('Time (s)', 'FontSize', fontSize, 'Interpreter', 'latex');
ylabel('$\dot{f}_a$ (m$^3$/s)', 'FontSize', fontSize, 'Interpreter', 'latex');
title('Control Input: Air Mass Flow Rate ($\dot{f}_a$)', 'FontSize', fontSize, 'Interpreter', 'latex');
set(gca, 'FontSize', fontSize, 'LineWidth', 1);

subplot(3,1,2);
plot(Time, fw_dot, 'b', 'LineWidth', lineWidth);
grid on;
xlabel('Time (s)', 'FontSize', fontSize, 'Interpreter', 'latex');
ylabel('$\dot{f}_w$ (m$^3$/s)', 'FontSize', fontSize, 'Interpreter', 'latex');
title('Control Input: Chilled Water Flow Rate ($\dot{f}_w$)', 'FontSize', fontSize, 'Interpreter', 'latex');
set(gca, 'FontSize', fontSize, 'LineWidth', 1);

subplot(3,1,3);
plot(Time, u3, 'g', 'LineWidth', lineWidth);
grid on;
xlabel('Time (s)', 'FontSize', fontSize, 'Interpreter', 'latex');
ylabel('$\dot{m}_{net,moist}$ (g/m$^3$)', 'FontSize', fontSize, 'Interpreter', 'latex');
title('Control Input: Net Moisture Addition/Removal ($\dot{m}_{net,moist}$)', 'FontSize', fontSize, 'Interpreter', 'latex');
set(gca, 'FontSize', fontSize, 'LineWidth', 1);

set(gcf, 'PaperPositionMode', 'auto'); % Ensures better layout for printing

%% Plot Time Response with Reference Tracking
figure;
set(gcf, 'Position', [100, 100, 600, 500]); % Adjust figure size

subplot(3,1,1);
plot(Time, Tt_actual, 'b', 'LineWidth', lineWidth);
hold on;
plot(Time, Tt_ref, '--r', 'LineWidth', lineWidth);
grid on;
xlabel('Time (s)', 'FontSize', fontSize, 'Interpreter', 'latex');
ylabel('$T_t$ (°C)', 'FontSize', fontSize, 'Interpreter', 'latex');
legend({'$T_t$ actual', '$T_t$ ref'}, 'FontSize', fontSize, 'Interpreter', 'latex', 'Location', 'Best');
title('Zone Temperature ($T_t$)', 'FontSize', fontSize, 'Interpreter', 'latex');
set(gca, 'FontSize', fontSize, 'LineWidth', 1);

subplot(3,1,2);
plot(Time, wt_actual, 'b', 'LineWidth', lineWidth);
hold on;
plot(Time, wt_ref, '--r', 'LineWidth', lineWidth);
grid on;
xlabel('Time (s)', 'FontSize', fontSize, 'Interpreter', 'latex');
ylabel('$w_t$', 'FontSize', fontSize, 'Interpreter', 'latex');
legend({'$w_t$ actual', '$w_t$ ref'}, 'FontSize', fontSize, 'Interpreter', 'latex', 'Location', 'Best');
title('Zone Humidity ($w_t$)', 'FontSize', fontSize, 'Interpreter', 'latex');
set(gca, 'FontSize', fontSize, 'LineWidth', 1);

subplot(3,1,3);
plot(Time, Ts_actual, 'b', 'LineWidth', lineWidth);
hold on;
plot(Time, Ts_ref, '--r', 'LineWidth', lineWidth);
grid on;
xlabel('Time (s)', 'FontSize', fontSize, 'Interpreter', 'latex');
ylabel('$T_s$ (°C)', 'FontSize', fontSize, 'Interpreter', 'latex');
legend({'$T_s$ actual', '$T_s$ ref'}, 'FontSize', fontSize, 'Interpreter', 'latex', 'Location', 'Best');
title('Supply Air Temperature ($T_s$)', 'FontSize', fontSize, 'Interpreter', 'latex');
set(gca, 'FontSize', fontSize, 'LineWidth', 1);

set(gcf, 'PaperPositionMode', 'auto'); % Ensures better layout for printing

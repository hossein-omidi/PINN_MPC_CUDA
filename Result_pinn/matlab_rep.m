clc
clear
% Load the CSV file
data = readtable('mpc_simulation_20250401_175956.csv');

% Extract data
Time = data.Time;
Tt_actual = data.Tt_actual;
wt_actual = data.wt_actual;
Ts_actual = data.Ts_actual;
fa_dot = data.fa_dot;
fw_dot = data.fw_dot;
u3 = data.u3;
Tt_ref = data.Tt_ref;
wt_ref = data.wt_ref;
Ts_ref = data.Ts_ref;

% Plot Control Variables in Subplots
figure;
subplot(3,1,1);
hold on; grid on;
plot(Time, fa_dot, 'r', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('fa\_dot');
title('Control Input: fa\_dot');

subplot(3,1,2);
hold on; grid on;
plot(Time, fw_dot, 'b', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('fw\_dot');
title('Control Input: fw\_dot');

subplot(3,1,3);
hold on; grid on;
plot(Time, u3, 'g', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('u3');
title('Control Input: u3');

hold off;

% Plot Time Response with Reference Tracking in Subplots
figure;
subplot(3,1,1);
hold on; grid on;
plot(Time, Tt_actual, 'b', 'LineWidth', 1.5);
plot(Time, Tt_ref, '--r', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Tt (°C)');
legend('Tt\_actual', 'Tt\_ref', 'Location', 'Best');
title('Time Response: Tt Tracking');

subplot(3,1,2);
hold on; grid on;
plot(Time, wt_actual, 'b', 'LineWidth', 1.5);
plot(Time, wt_ref, '--r', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('wt');
legend('wt\_actual', 'wt\_ref', 'Location', 'Best');
title('Time Response: wt Tracking');

subplot(3,1,3);
hold on; grid on;
plot(Time, Ts_actual, 'b', 'LineWidth', 1.5);
plot(Time, Ts_ref, '--r', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Ts (°C)');
legend('Ts\_actual', 'Ts\_ref', 'Location', 'Best');
title('Time Response: Ts Tracking');

hold off;

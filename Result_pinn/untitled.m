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

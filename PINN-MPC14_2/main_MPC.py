from BldgEnergy import BuildingModel, schedule
from model import get_model
from MPC import cost_fun_min
from plotting import BldgEnergyPlot, MPCplot
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import csv
import os
import matplotlib

matplotlib.use('TkAgg')

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# import T_amb
with open('Toutdoor.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile))
t_amb = np.array(data).astype(float)
# time setting
dt = 1  # time interval [s]
endtime = 8 * 60 * 60 / dt  # 9:00-17:00
tevap, t_set, vdot_air, n_people = schedule(endtime)
# Bldg energy model generate
bldg = BuildingModel(dt=dt)
[t_room , power, ac_onoff] = bldg.datagen(n_people, tevap, t_set, vdot_air, t_amb)
# Plot BldgEnergy data
BldgEnergyPlot(t_room, t_amb, t_set, ac_onoff, tevap, power)
# NN/PINN Modeling
x_array = np.concatenate((t_room[:-1].reshape(-1, 1),
                          t_amb.reshape(-1, 1),
                          tevap.reshape(-1, 1),
                          vdot_air.reshape(-1, 1)), axis=1)  # shift for t_room(t+1)
y_array = t_room[1:].reshape(-1, 1)  # shift for t_room(t+1)
x_train, x_test, y_train, y_test = train_test_split(x_array, y_array, test_size=0.3, random_state=0)

x_train = x_train.astype('float64')
y_train = y_train.astype('float64')
x_train = torch.Tensor(x_train).to(device)
y_train = torch.Tensor(y_train).to(device)

x_test = x_test.astype('float64')
y_test = y_test.astype('float64')
x_test = torch.Tensor(x_test).to(device)
y_test = torch.Tensor(y_test).to(device)

# Model define
hidden_dim = 4
layer_dim = 2
input_dim = x_array.shape[1]
output_dim = y_array.shape[1]

model_params = {'input_dim': input_dim,
                'hidden_dim': round(hidden_dim),
                'layer_dim': round(layer_dim),
                'output_dim': output_dim}
model_name = "ann"
model = get_model(model_name, model_params)
model.load_state_dict(torch.load("PINN_STZ.pth", map_location=torch.device('cpu')))

model.to(device)
model.eval()

# Model Predictive Control
n = x_test.shape[0]  # total time step
# Initialization
u = np.array([x_test[0, 2].item()])  # control parameter: Tevap
y = np.array([x_test[0, 0].item()])  # actual room temperature
yp_int = np.array([x_test[0, 0].item()])  # predicted room temperature
E_int = 1200/15 * 25-(np.array([x_test[0, 2].item()]))  # power
# on-line control
for i in range(1, n):
    # forecasting & schedule
    t_amb = np.array([x_test[i, 1].item()])  # T_amb
    cfmtoms = 0.00047194745
    vdot_air = 424 * cfmtoms  # air flow rate
    n_people = 5  # occupancy
    ym = t_set  # reference temperature
    # MPC algo setting
    upp = 20  # control para. upper bound
    low = 10  # control para. lower bound
    # MPC algo operating
    u_next, yp, E_cost = cost_fun_min(t_amb[-1], u[-1], y[-1], ym[-1], upp, low, model)
    u_next = np.squeeze(u_next, axis=(2,))
    E_cost = np.squeeze(E_cost, axis=(2,))
    yp = np.squeeze(yp, axis=(1,))
    print(u_next[0])
    print(yp[0])
    bldg = BuildingModel(dt=dt)
    y_act = bldg.testbed(n_people, u_next[0], y[-1], vdot_air, t_amb)  # actual room temperature
    # Data log
    y = np.concatenate((y, y_act), axis=0)
    u = np.concatenate((u.reshape(-1, 1), u_next), axis=0)
    yp_int = np.concatenate((yp_int, yp), axis=0)
    E_int = np.concatenate((E_int.reshape(-1, 1), E_cost), axis=0)
# MPC plotting
MPCplot(y, yp_int, u, E_int)
E_onoff = np.sum(power[x_train.shape[0]:]) / 1000 / 60
print("\nEnergy Consumption (baseline) [kWh]: ", E_onoff)

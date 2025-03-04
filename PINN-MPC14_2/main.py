from BldgEnergy import BuildingModel, schedule
from model import get_model, Optimization
from plotting import BldgEnergyPlot, testplot
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.optim as optim
import csv
import os
import matplotlib

matplotlib.use('TkAgg')

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# setting
mo = 'PINN'
# import T_amb profile
with open('Toutdoor.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile))
t_amb = np.array(data).astype(float)
# time setting
dt = 60  # time interval [s]
endtime = 8 * 60 * 60 / dt  # 9:00-17:00
tevap, t_set, vdot_air, n_people = schedule(endtime)
# Bldg energy model generate
bldg = BuildingModel(dt=dt)
[t_room, power, ac_onoff] = bldg.datagen(n_people, tevap, t_set, vdot_air, t_amb)
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
model = model.to(device)
# optimizer setup
learning_rate = 1e-3
weight_decay = 1e-6
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
opt = Optimization(model=model, mo=mo, optimizer=optimizer)
# training
n_epochs = 20000
opt.train(x_train, y_train, n_epochs)
# Save model
torch.save(model.state_dict(), mo + "_STZ.pth")
# Testing Prediction
yt = model(x_test)
loss_test = torch.mean((yt-y_test)**2).detach()  # MSE test
print('MSE of test: {:.10f}'.format(loss_test))
# Model Test Plot
yact = yt.cpu().detach().numpy()
ypred = y_test.cpu().reshape(-1, 1)
testplot(yact, ypred)



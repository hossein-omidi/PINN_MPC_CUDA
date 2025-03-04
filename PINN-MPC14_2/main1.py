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
from model import get_model, Optimization
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.optim as optim
import pandas as pd
import matplotlib



print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("Device count:", torch.cuda.device_count())
print("Current device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(0))


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
controls_df = pd.read_csv('mpc_controls.csv')
inputs = controls_df[['MPC fa_dot', 'MPC fw_dot', 'MPC u3']].values
inputs[:, 2] = inputs[:, 2] * 1000  # Multiply 'MPC u3' column by 1000

# Load state outputs
states_df = pd.read_csv('mpc_states.csv')
outputs = states_df[['MPC Tt', 'MPC wt', 'MPC Ts']].values
outputs[:, 1] = outputs[:, 1] * 1000  # Multiply 'MPC wt' column by 1000

# Create sequence data (current state + current input -> next state)
x_array = np.hstack((outputs[:-1], inputs[:-1]))  # Shape: (N-1, 6)
y_array = outputs[1:]                             # Shape: (N-1, 3)

# Train-test split
# Replace with time-series-friendly split:
test_size = 0.3
split_idx = int(len(x_array) * (1 - test_size))
x_train, x_test = x_array[:split_idx], x_array[split_idx:]
y_train, y_test = y_array[:split_idx], y_array[split_idx:]
# Convert to tensors
x_train = torch.FloatTensor(x_train).to(device)
y_train = torch.FloatTensor(y_train).to(device)
x_test = torch.FloatTensor(x_test).to(device)
y_test = torch.FloatTensor(y_test).to(device)

# Model define
mo = 'PINN'
hidden_dim = 256  # Increased capacity for MIMO system
layer_dim = 8   # Deeper network
input_dim = 6      # 3 states + 3 inputs
output_dim = 3     # 3 output states

model_params = {
    'input_dim': input_dim,
    'hidden_dim': hidden_dim,
    'layer_dim': layer_dim,
    'output_dim': output_dim
}
model_name = "lstm"
model = get_model(model_name, model_params)
model = model.to(device)
# optimizer setup
learning_rate = 1e-4
weight_decay = 1e-6
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
opt = Optimization(model=model, mo=mo, optimizer=optimizer)
# training
n_epochs = 5000
opt.train(x_train, y_train, n_epochs)
# Save model
torch.save(model.state_dict(), mo + "_STZ.pth")
# Testing Prediction
yt = model(x_test)
loss_test = torch.mean((yt - y_test) ** 2).detach()  # MSE test
print('MSE of test: {:.10f}'.format(loss_test))

# Model Test Plot
yact = yt.cpu().detach().numpy()  # Convert to numpy
ypred = y_test.cpu().reshape(-1, 3).numpy()  # Ensure ypred is reshaped to (N, 3)

testplot(yact, ypred)




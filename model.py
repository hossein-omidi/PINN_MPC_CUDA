import torch
import torch.nn as nn
import torch.nn.functional as F
import time

seed = 42
torch.manual_seed(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Parameters
# Constants and parameters
w0 = 8.2      # g/kg, initial humidity ratio of outdoor air (originally 0.0082 kg/kg * 1000)
ws = 8.0      # g/kg, constant humidity ratio of supply air (originally 0.0080 kg/kg * 1000)
Cpa = 1000      # J/(kg*K), specific heat capacity of dry air
Cpw = 4180      # J/(kg*K), specific heat capacity of water vapor
hw = 800 * 1000 # J/kg, enthalpy of vaporization of water
hfg = 2500 * 1000 # J/kg, latent heat of vaporization of water
rho_a = 1.18    # kg/m^3, air density
rho_w = 1000    # kg/m^3, water density
Vt = 400        # m^3, zone volume
Vc = 1          # m^3, supply air volume
fa_dot0 = 2.6   # m^3/s, initial supply air volume flow rate
fw_dot0 = 0.9 / 1000 # kg/s, initial supply water mass flow rate
T0 = 32         # degC, outdoor air temperature
dTc = 6         # degC, cooling coil temperature difference
M_dot0 = 0.000115 # kg/s, initial moisture transfer rate
Q_dot0 = 20 * 1000 # W, cooling coil heat transfer rate

# Derived parameters
alpha1 = 1 / Vt
alpha2 = 1 / (rho_a * Vt)
alpha3 = 1 / Vc
beta1 = hfg / (Cpa * Vt)
beta2 = (rho_w * Cpw * dTc) / (rho_a * Cpa * Vc)
gamma1 = 1 / (rho_a * Cpa * Vt)
gamma2 = hw / (Cpa * Vc)


class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = layer_dim
        self.output_dim = output_dim
        # self.dropout = dropout_prob
        self.flatten = nn.Flatten()
        self.layers = nn.ModuleDict()
        self.layers["input"] = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        for i in range(self.n_layers):
            self.layers[f"hidden_{i}"] = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
            # self.layers[f"dropout{i}"] = nn.Dropout(dropout_prob)
        self.layers["output"] = nn.Linear(in_features=hidden_dim, out_features=output_dim)

    def forward(self, x):
        x = self.layers["input"](self.flatten(x))
        for i in range(self.n_layers):
            x = F.tanh(self.layers[f"hidden_{i}"](x))  # tanh/relu
            # x = self.layers[f"dropout{i}"](x)
        return self.layers["output"](x)


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.residual = nn.Linear(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Ensure x is 3D (batch_size, seq_length, input_dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension if missing

        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_length, hidden_dim)

        # Residual connection
        res = self.residual(x[:, -1, :])  # (batch_size, hidden_dim)

        # Combine LSTM and residual output
        out = lstm_out[:, -1, :] + res

        return self.fc(out)  # Output shape: (batch_size, output_dim)


# Update get_model function
def get_model(model, model_params):
    models = {
        "ann": NeuralNetwork,
        "lstm": LSTMModel  # Add this line
    }
    return models.get(model.lower())(**model_params)


class LossFuc:
    def __init__(self, model, x_train, y_train):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train

    def nnloss(self):
        yh = self.model(self.x_train)
        # In LossFuc.nnloss, modify:
        loss = 0.3 * torch.mean((yh[:, 0] - self.y_train[:, 0]) ** 2) + \
               1.0 * torch.mean((yh[:, 1] - self.y_train[:, 1]) ** 2) + \
               0.3 * torch.mean((yh[:, 2] - self.y_train[:, 2]) ** 2)
        return loss

    def pinnloss(self, dt, lamda):
        yh = self.model(self.x_train)  # Predicted next states [Tt_next, wt_next, Ts_next]
        loss1 = torch.mean((yh[:, 0] - self.y_train[:, 0]) ** 2) + \
                1.5*torch.mean((yh[:, 1] - self.y_train[:, 1]) ** 2) + \
                torch.mean((yh[:, 2] - self.y_train[:, 2]) ** 2)

        # Physics residual calculation
        x_pinn = self.x_train.requires_grad_(True)
        Tt, wt, Ts = x_pinn[:, 0], x_pinn[:, 1], x_pinn[:, 2]
        fa_dot, fw_dot, u3 = x_pinn[:, 3], x_pinn[:, 4], x_pinn[:, 5]

        # Improved Predicted derivatives (discrete-time) using central difference method
        # Equations
        # Improved Predicted derivatives
        # Differential equations
        dTt = (1 / (rho_a * Cpa * Vt)) * (Q_dot0 - hfg * M_dot0) + \
              ((fa_dot * hfg) / (1000 * Cpa * Vt)) * (wt - ws) - (fa_dot / Vt) * (Tt - Ts)

        dwt = 1000 * (M_dot0 / (rho_a * Vt)) - (fa_dot / Vt) * (wt - ws) + u3

        dTs = (fa_dot / Vc) * (Tt - Ts) + \
              (0.25 * fa_dot / Vc) * (T0 - Tt) - \
              (fa_dot * hw / (1000 * Cpa * Vc)) * (0.25 * w0 + 0.75 * wt - ws) - \
              (fw_dot * rho_w * Cpw * dTc) / (rho_a * Cpa * Vc)


        Tt_next, wt_next, Ts_next = yh[:, 0], yh[:, 1], yh[:, 2]
        Tt_prev, wt_prev, Ts_prev = Tt - dTt * dt, wt - dwt * dt, Ts - dTs * dt
        dTt_pred = (Tt_next - Tt_prev) / (2 * dt)
        dwt_pred = (wt_next - wt_prev) / (2 * dt)
        dTs_pred = (Ts_next - Ts_prev) / (2 * dt)

        # Residuals
        physics_loss = (
                torch.mean((dTt_pred - dTt) ** 2) +
                torch.mean((dwt_pred - dwt) ** 2) +
                torch.mean((dTs_pred - dTs) ** 2)
        )

        total_loss = loss1 + lamda * physics_loss

        # Add penalty for extreme values
        # In LossFuc.pinnloss, replace:
        penalty = torch.mean(torch.relu(yh[:, 0] - 35))  # Only penalize Tt
        return total_loss + 0.1 * penalty  # Adjust weight


class Optimization:
    def __init__(self, model, mo, optimizer):
        self.model = model
        self.mo = mo
        self.optimizer = optimizer

    def train(self, x_train, y_train, n_epochs):
        if self.mo == 'NN':
            start = time.time()
            print("start")
            for i in range(n_epochs):
                self.optimizer.zero_grad()
                nnlossfuc = LossFuc(model=self.model, x_train=x_train, y_train=y_train)
                loss = nnlossfuc.nnloss()
                loss.backward()
                self.optimizer.step()
                if (i + 1) % 100 == 0:
                    print('Epoch [{:10}/{}], Loss: {:.10f}'.format(i + 1, n_epochs, loss.item()))
            end = time.time()
            print(end - start)

        if self.mo == 'PINN':
            start = time.time()  # ✅ Added this line to fix the error
            for i in range(n_epochs):
                dt = .5  # Match your timestep
                self.optimizer.zero_grad()
                initial_lambda = 1e-1
                scaling_factor = 1 + i / n_epochs  # Gradually increase lambda

                lamda = 5 # Gradually increase physics weight

                # Compute PINN loss with new physics
                nnlossfuc = LossFuc(model=self.model, x_train=x_train, y_train=y_train)
                loss = nnlossfuc.pinnloss(dt=dt, lamda=lamda)

                loss.backward()
                self.optimizer.step()
                if (i + 1) % 100 == 0:
                    print('Epoch [{:10}/{}], Loss: {:.10f}'.format(i + 1, n_epochs, loss.item()))

            end = time.time()
            print(end - start)  # ✅ Now this will work correctly
        loss_train = loss.detach()  # MSE train
        print('MSE of train: {:.10f}'.format(loss_train))

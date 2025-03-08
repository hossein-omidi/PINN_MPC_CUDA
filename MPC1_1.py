import torch
import numpy as np
from scipy.optimize import minimize

class MPCSolver:
    def __init__(self, model, bounds, params):
        self.model = model
        self.bounds = bounds
        self.params = params
        self.prev_solution = None
        
        # Control bounds with 2% safety margin
        self.control_lb = np.array([bounds['control'][k][0] for k in ['fa_dot','fw_dot','u3']], 
                                  dtype=np.float32) * 1.02
        self.control_ub = np.array([bounds['control'][k][1] for k in ['fa_dot','fw_dot','u3']], 
                                  dtype=np.float32) * 0.98
        
        # State bounds with Ts priority
        self.state_bounds = np.array([
            [bounds['state']['Tt'][0], bounds['state']['Tt'][1]],
            [bounds['state']['wt'][0], bounds['state']['wt'][1]],
            [max(bounds['state']['Ts'][0], 12.0), min(bounds['state']['Ts'][1], 24.0)]
        ], dtype=np.float32)

    def solve(self, current_states, prev_controls, references):
        horizon = self.params['horizon']
        
        # Enhanced warm-start with momentum and noise rejection
        if self.prev_solution is not None:
            controls_sequence = 0.6 * np.roll(self.prev_solution, -2, axis=0)
            controls_sequence += 0.4 * np.tile(prev_controls, (horizon, 1))
            controls_sequence = np.clip(controls_sequence, self.control_lb, self.control_ub)
        else:
            controls_sequence = np.tile(prev_controls, (horizon, 1))
        
        # Optimization with improved stability settings
        result = minimize(
            self._objective,
            controls_sequence.flatten(),
            args=(current_states, references),
            method='L-BFGS-B',
            jac=True,
            bounds=[(lb, ub) for lb, ub in zip(
                np.tile(self.control_lb, horizon),
                np.tile(self.control_ub, horizon)
            )],
            options={
                'maxiter': self.params['max_iter'],
                'ftol': 1e-9,
                'gtol': 1e-7,
                'maxcor': 20
            }
        )
        
        self.prev_solution = result.x.reshape(horizon, 3)
        return self.prev_solution[0], None, 0

    def _objective(self, u_flat, current_states, references):
        device = next(self.model.parameters()).device
        u = u_flat.reshape(self.params['horizon'], 3).astype(np.float32)
        state = torch.tensor(current_states, dtype=torch.float32, device=device)
        total_cost = 0.0
        grad = np.zeros_like(u)
        
        for t in range(self.params['horizon']):
            # Efficient state prediction
            with torch.enable_grad():
                control_tensor = torch.tensor(u[t], device=device, requires_grad=True)
                nn_input = torch.cat([state, control_tensor])
                pred = self.model(nn_input.unsqueeze(0)).squeeze()
                
                # Full Jacobian calculation
                jac = torch.autograd.functional.jacobian(
                    lambda x: self.model(x.unsqueeze(0)), nn_input
                )[0]  # Shape: [3, 6]

            # Extract control Jacobian (last 3 columns)
            jac_control = jac[:, 3:].detach().cpu().numpy()
            
            # State predictions
            next_state = pred.detach().cpu().numpy().astype(np.float32)
            
            # Enhanced constraint handling
            state_violation = np.maximum(
                next_state - self.state_bounds[:, 1],
                self.state_bounds[:, 0] - next_state
            )
            state_violation = np.maximum(state_violation, 0)
            
            # Adaptive penalty for Ts violations
            ts_violation = state_violation[2]
            if ts_violation > 0:
                state_violation[2] *= 10 * np.exp(2*ts_violation)
            
            # Cost components
            tracking_error = next_state - references
            cost_track = np.dot(tracking_error, self.params['W'] @ tracking_error)
            cost_con = np.dot(state_violation, self.params['S'] @ state_violation)
            cost_control = np.dot(u[t], self.params['R'] @ u[t])
            cost_smooth = 0.1 * np.sum(np.diff(u[max(0,t-1):t+1], axis=0)**2) if t > 0 else 0
            
            total_cost += (
                self.params['lambda_tracking'] * cost_track +
                self.params['lambda_con'] * cost_con +
                cost_control +
                cost_smooth
            )
            
            # Gradient calculations
            grad_track = 2 * jac_control.T @ (self.params['W'] @ tracking_error)
            grad_con = 2 * jac_control.T @ (self.params['S'] @ state_violation)
            grad_control = 2 * (self.params['R'] @ u[t])
            grad_smooth = 0.2 * (u[t] - u[t-1]) if t > 0 else 0
            
            grad[t] = grad_track + grad_con + grad_control + grad_smooth
            state = torch.tensor(next_state, dtype=torch.float32, device=device)
        
        return total_cost.astype(np.float64), grad.flatten().astype(np.float64)

def cost_fun_mimo(current_states, prev_controls, references, bounds, model,
                  W, R, lambda_tracking, lambda_terminal, lambda_integral,
                  w_state_con, w_control_con, s, horizon, dt, max_iter):
    
    params = {
        'W': W.astype(np.float32),
        'R': R.astype(np.float32),
        'S': np.diag([5e4, 5e4, 1e5]).astype(np.float32),  # High Ts constraint weight
        'lambda_tracking': np.float32(0.9),
        'lambda_con': np.float32(10.0),  # Increased constraint weight
        'horizon': min(horizon, 30),  # Optimal horizon length
        'dt': dt,
        'max_iter': max_iter
    }
    
    return MPCSolver(model, bounds, params).solve(
        current_states.astype(np.float32),
        prev_controls.astype(np.float32),
        references.astype(np.float32)
    )
import torch
import numpy as np
from scipy.optimize import minimize

class MPCSolver:
    def __init__(self, model, bounds, params):
        self.model = model
        self.bounds = bounds
        self.params = params
        self.prev_solution = None
        
        # Control bounds with safety buffer
        self.control_lb = np.array([bounds['control'][k][0] for k in ['fa_dot','fw_dot','u3']], 
                                  dtype=np.float32) + 1e-3
        self.control_ub = np.array([bounds['control'][k][1] for k in ['fa_dot','fw_dot','u3']], 
                                  dtype=np.float32) - 1e-3
        
        # State bounds with correct dimensions
        self.state_bounds = np.array([
            [bounds['state']['Tt'][0], bounds['state']['Tt'][1]],
            [bounds['state']['wt'][0], bounds['state']['wt'][1]],
            [bounds['state']['Ts'][0], bounds['state']['Ts'][1]]
        ], dtype=np.float32)

    def solve(self, current_states, prev_controls, references):
        # Stable warm-start initialization
        horizon = self.params['horizon']
        if self.prev_solution is not None and len(self.prev_solution) == horizon:
            controls_sequence = 0.8 * np.roll(self.prev_solution, -2, axis=0)
            controls_sequence += 0.2 * np.random.normal(0, 0.05, size=(horizon, 3))
        else:
            controls_sequence = np.tile(prev_controls, (horizon, 1))
        
        # Bounds-safe initialization
        controls_sequence = np.clip(controls_sequence, self.control_lb, self.control_ub)
        
        # Optimization with stability constraints
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
                'ftol': 1e-8,
                'gtol': 1e-6
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
            # State prediction
            control_tensor = torch.tensor(u[t], device=device, dtype=torch.float32)
            nn_input = torch.cat([state, control_tensor])
            with torch.enable_grad():
                nn_input.requires_grad_(True)
                pred = self.model(nn_input.unsqueeze(0)).squeeze()
            
            # Gradient calculation
            jac = torch.autograd.functional.jacobian(
                lambda x: self.model(x.unsqueeze(0)), nn_input
            )[0, :, 3:].detach().cpu().numpy().astype(np.float32)
            
            next_state = pred.detach().cpu().numpy().astype(np.float32)
            
            # State constraint handling
            upper_violation = next_state - self.state_bounds[:, 1]
            lower_violation = self.state_bounds[:, 0] - next_state
            state_violation = np.maximum(np.maximum(upper_violation, lower_violation), 0)
            
            # Cost components
            tracking_error = next_state - references
            cost_track = np.dot(tracking_error, self.params['W'] @ tracking_error)
            cost_con = np.dot(state_violation, self.params['S'] @ state_violation)
            cost_control = np.dot(u[t], self.params['R'] @ u[t])
            
            # Gradient calculation
            grad_track = 2 * (jac.T @ (self.params['W'] @ tracking_error))
            grad_con = 2 * (jac.T @ (self.params['S'] @ state_violation))
            grad_control = 2 * (self.params['R'] @ u[t])
            
            total_cost += self.params['lambda_tracking'] * cost_track \
                        + self.params['lambda_con'] * cost_con \
                        + cost_control
            
            grad[t] = grad_track + grad_con + grad_control
            state = torch.tensor(next_state, dtype=torch.float32, device=device)
        
        return total_cost.astype(np.float64), grad.flatten().astype(np.float64)

def cost_fun_mimo(current_states, prev_controls, references, bounds, model,
                  W, R, lambda_tracking, lambda_terminal, lambda_integral,
                  w_state_con, w_control_con, s, horizon, dt, max_iter):
    
    params = {
        'W': W.astype(np.float32),
        'R': R.astype(np.float32),
        'S': np.diag([1e4, 1e4, 1e2]).astype(np.float32),  # State constraint weights
        'lambda_tracking': np.float32(lambda_tracking),
        'lambda_con': np.float32(1.0),  # Constraint penalty weight
        'horizon': min(horizon, 30),  # Stability-focused horizon
        'dt': dt,
        'max_iter': max_iter
    }
    
    return MPCSolver(model, bounds, params).solve(
        current_states.astype(np.float32),
        prev_controls.astype(np.float32),
        references.astype(np.float32)
    )
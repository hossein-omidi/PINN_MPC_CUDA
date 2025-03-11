import torch
import numpy as np
from scipy.optimize import minimize

class MPCSolver:
    def __init__(self, model, bounds, params):
        self.model = model
        self.bounds = bounds
        self.params = params
        self.prev_solution = None
        self.integral_error = np.zeros(3, dtype=np.float32)
        
        # Initialize control bounds
        self.control_lb = np.array([bounds['control'][k][0] for k in ['fa_dot','fw_dot','u3']], 
                                  dtype=np.float32)
        self.control_ub = np.array([bounds['control'][k][1] for k in ['fa_dot','fw_dot','u3']], 
                                  dtype=np.float32)
        self.state_bounds = np.array([[bounds['state'][k][0], bounds['state'][k][1]] 
                                    for k in ['Tt','wt','Ts']], dtype=np.float32)

    def solve(self, current_states, prev_controls, references):
        
        # Warm-start initialization with momentum
        if self.prev_solution is not None:
            controls_sequence = 0.7 * np.roll(self.prev_solution, -1, axis=0) + \
                              0.3 * np.tile(prev_controls, (self.params['horizon'], 1))
            controls_sequence[-1] = controls_sequence[-2]
        else:
            controls_sequence = np.tile(prev_controls, (self.params['horizon'], 1))
        
        
        # Optimization using L-BFGS-B with bounds
        result = minimize(
            self._objective,
            controls_sequence.flatten(),
            args=(current_states, references),
            method='L-BFGS-B',
            jac=True,
            bounds=[(lb, ub) for lb, ub in zip(
                np.tile(self.control_lb, self.params['horizon']),
                np.tile(self.control_ub, self.params['horizon'])
            )],
            options={
                'maxiter': self.params['max_iter'],
                'ftol': 1e-6,
                'disp': False
            }
        )
        
        self.prev_solution = result.x.reshape(self.params['horizon'], 3)        
        return result.x[:3], None, 0

    def _objective(self, u_flat, current_states, references):
        device = next(self.model.parameters()).device
        u = u_flat.reshape(self.params['horizon'], 3).astype(np.float32)
        state = torch.tensor(current_states, dtype=torch.float32, device=device)
        total_cost = 0.0
        grad = np.zeros_like(u)
                
        for t in range(self.params['horizon']):            
            # Neural network prediction
            control_tensor = torch.tensor(u[t], device=device, dtype=torch.float32)
            nn_input = torch.cat([state, control_tensor])
            nn_input.requires_grad = True
            pred = self.model(nn_input.unsqueeze(0)).squeeze()
                        
            # Calculate gradients using automatic differentiation
            jac = torch.autograd.functional.jacobian(
                lambda x: self.model(x.unsqueeze(0)), 
                nn_input
            )[0, :, 3:].detach().cpu().numpy().astype(np.float32)
            
            # State predictions
            next_state = pred.detach().cpu().numpy().astype(np.float32)
            
            # Tracking calculations
            tracking_error = next_state - references
            self.integral_error += tracking_error * self.params['dt']
                        
            # Cost components
            cost_track = self.params['lambda_tracking'] * np.dot(tracking_error, self.params['W'] @ tracking_error)
            cost_control = np.dot(u[t], self.params['R'] @ u[t])
            cost_terminal = self.params['lambda_terminal'] * np.dot(tracking_error, tracking_error) if t == self.params['horizon']-1 else 0
                        
            # Gradient calculations
            grad_track = 2 * self.params['lambda_tracking'] * (jac.T @ (self.params['W'] @ tracking_error))
            grad_control = 2 * (self.params['R'] @ u[t])
            
            total_cost += cost_track + cost_control + cost_terminal
            grad[t] = grad_track + grad_control
            
            
            state = torch.tensor(next_state, dtype=torch.float32, device=device)
        
        return total_cost.astype(np.float64), grad.flatten().astype(np.float64)

def cost_fun_mimo(current_states, prev_controls, references, bounds, model,
                  W, R, lambda_tracking, lambda_terminal, lambda_integral,
                  w_state_con, w_control_con, s, horizon, dt, max_iter):
    
    params = {
        'W': W.astype(np.float32),
        'R': R.astype(np.float32),
        'lambda_tracking': np.float32(lambda_tracking),
        'lambda_terminal': np.float32(lambda_terminal),
        'horizon': horizon,
        'dt': dt,
        'max_iter': max_iter
    }
    
    solver = MPCSolver(model, bounds, params)
    
    return solver.solve(
        current_states.astype(np.float32),
        prev_controls.astype(np.float32),
        references.astype(np.float32)
    )
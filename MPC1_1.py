import torch
import numpy as np
from scipy.optimize import minimize

class MPCSolver:
    def __init__(self, model, bounds, params):
        self.model = model
        self.bounds = bounds
        self.params = params
        self.prev_solution = None
        
        # Control bounds with slight relaxation for flexibility
        self.control_lb = np.array([bounds['control'][k][0] for k in ['fa_dot', 'fw_dot', 'u3']], 
                                  dtype=np.float32) + 1e-4
        self.control_ub = np.array([bounds['control'][k][1] for k in ['fa_dot', 'fw_dot', 'u3']], 
                                  dtype=np.float32) - 1e-4
        
        # Tighter state bounds for physical realism
        self.state_bounds = np.array([
            [max(bounds['state']['Tt'][0], 15.0), min(bounds['state']['Tt'][1], 32.0)],
            [max(bounds['state']['wt'][0], 6.0), min(bounds['state']['wt'][1], 9.5)],
            [max(bounds['state']['Ts'][0], 12.0), min(bounds['state']['Ts'][1], 25.0)]
        ], dtype=np.float32)

    def solve(self, current_states, prev_controls, references):
        horizon = self.params['horizon']
        
        # NaN-safe warm-start with more emphasis on previous solution
        if self.prev_solution is not None and not np.any(np.isnan(self.prev_solution)):
            controls_sequence = 0.8 * np.roll(self.prev_solution, -1, axis=0) + \
                               0.2 * np.tile(prev_controls, (horizon, 1))
        else:
            controls_sequence = np.tile(prev_controls, (horizon, 1))
        
        controls_sequence = np.clip(np.nan_to_num(controls_sequence), 
                                  self.control_lb, self.control_ub)
        
        # Optimization with tuned parameters
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
                'ftol': 1e-7,  # Tighter tolerance for precision
                'gtol': 1e-5,  # Increased gradient tolerance
                'maxfun': 1000,  # More function evaluations
                'maxls': 100   # More line search steps
            }
        )
        
        self.prev_solution = np.nan_to_num(result.x.reshape(horizon, 3))
        return self.prev_solution[0], None, 0

    def _objective(self, u_flat, current_states, references):
        device = next(self.model.parameters()).device
        u = u_flat.reshape(self.params['horizon'], 3).astype(np.float32)
        state = torch.tensor(current_states, dtype=torch.float32, device=device)
        total_cost = 0.0
        grad = np.zeros_like(u)
        
        for t in range(self.params['horizon']):
            try:
                u_t = np.nan_to_num(u[t])
                control_tensor = torch.tensor(u_t, device=device, requires_grad=True)
                
                with torch.enable_grad():
                    nn_input = torch.cat([state, control_tensor])
                    if torch.isnan(nn_input).any():
                        raise RuntimeError("NaN detected in NN input")
                    
                    pred = self.model(nn_input.unsqueeze(0)).squeeze()
                    jac = torch.autograd.grad(
                        outputs=pred,
                        inputs=control_tensor,
                        grad_outputs=torch.ones_like(pred),
                        create_graph=False,
                        retain_graph=False,
                        allow_unused=True
                    )[0]
                
                jac_control = jac.detach().cpu().numpy().astype(np.float32) if jac is not None else np.zeros(3)
                
                next_state = pred.detach().cpu().numpy().astype(np.float32)
                next_state = np.nan_to_num(next_state)
                next_state = np.clip(next_state, self.state_bounds[:, 0], self.state_bounds[:, 1])
                
                # Enhanced constraint handling with tighter bounds
                state_violation = np.maximum(
                    next_state - self.state_bounds[:, 1],
                    self.state_bounds[:, 0] - next_state
                )
                state_violation = np.maximum(state_violation, 0)
                
                # Cost components with improved weighting
                tracking_error = next_state - references
                cost_track = np.dot(tracking_error, self.params['W'] @ tracking_error)
                cost_con = np.dot(state_violation, self.params['S'] @ state_violation)
                cost_control = np.dot(u_t, self.params['R'] @ u_t)
                
                # Smoothing term to encourage control variation
                cost_smooth = 0.0
                if t > 0:
                    control_diff = u_t - u[t-1]
                    cost_smooth = 0.1 * np.dot(control_diff, control_diff)
                
                total_cost += (
                    self.params['lambda_tracking'] * cost_track +
                    self.params['lambda_con'] * cost_con +
                    cost_control +
                    cost_smooth
                )
                
                # Gradient with increased sensitivity
                grad_track = 2 * (jac_control.T @ (self.params['W'] @ tracking_error))
                grad_con = 2 * (jac_control.T @ (self.params['S'] @ state_violation))
                grad_control = 2 * (self.params['R'] @ u_t)
                grad_smooth = 0.2 * (u_t - u[t-1]) if t > 0 else 0.0
                
                grad[t] = np.nan_to_num(grad_track + grad_con + grad_control + grad_smooth)
                state = torch.tensor(next_state, dtype=torch.float32, device=device)
                
            except Exception as e:
                print(f"Step {t} error: {str(e)}")
                grad[t] = np.zeros_like(grad[t])
                total_cost += 1e6
                break
        
        return np.nan_to_num(total_cost).astype(np.float64), grad.flatten().astype(np.float64)

def cost_fun_mimo(current_states, prev_controls, references, bounds, model,
                  W, R, lambda_tracking, lambda_terminal, lambda_integral,
                  w_state_con, w_control_con, s, horizon, dt, max_iter):
    
    params = {
        'W': np.diag([50.0, 50.0, 100.0]).astype(np.float32),  # Stronger tracking, especially for Ts
        'R': np.diag([1.0, 2.0, 1.0]).astype(np.float32),      # Reduced control penalty
        'S': np.diag([5e5, 5e5, 1e6]).astype(np.float32),      # Stronger Ts constraint
        'lambda_tracking': np.float32(1.0),                    # Full emphasis on tracking
        'lambda_con': np.float32(20.0),                        # Stronger constraint enforcement
        'horizon': min(horizon, 15),                           # Shorter horizon for faster response
        'dt': dt,
        'max_iter': max_iter
    }
    
    return MPCSolver(model, bounds, params).solve(
        np.nan_to_num(current_states.astype(np.float32)),
        np.nan_to_num(prev_controls.astype(np.float32)),
        np.nan_to_num(references.astype(np.float32))
    )
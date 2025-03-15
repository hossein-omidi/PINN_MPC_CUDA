import torch
import numpy as np
from scipy.optimize import minimize

class UnifiedMPCSolver:
    def __init__(self, model, bounds, params):
        """Initialize the MPC solver with the model, bounds, and parameters."""
        self.model = model.to(torch.float32)
        self.device = next(model.parameters()).device
        self.bounds = bounds
        self.params = params
        self.prev_solution = None
        self.integral_error = np.zeros(3, dtype=np.float32)
        
        self.control_lb = np.array([bounds['control'][k][0] for k in ['fa_dot', 'fw_dot', 'u3']], 
                                 dtype=np.float32)
        self.control_ub = np.array([bounds['control'][k][1] for k in ['fa_dot', 'fw_dot', 'u3']], 
                                 dtype=np.float32)
        self.state_bounds = np.array([
            [bounds['state']['Tt'][0], bounds['state']['Tt'][1]],
            [bounds['state']['wt'][0], bounds['state']['wt'][1]],
            [bounds['state']['Ts'][0], bounds['state']['Ts'][1]]
        ], dtype=np.float32)

    def solve(self, current_states, prev_controls, references):
        """Solve the MPC optimization problem."""
        horizon = self.params['horizon']
        
        if self.prev_solution is not None and not np.any(np.isnan(self.prev_solution)):
            controls_sequence = 0.9 * np.roll(self.prev_solution, -1, axis=0) + \
                               0.1 * np.tile(prev_controls, (horizon, 1))
            controls_sequence[-1] = controls_sequence[-2]
        else:
            controls_sequence = np.tile(prev_controls, (horizon, 1))
            if not hasattr(self, '_first_call'):
                print("No previous solution found. Using initial controls (first call).")
                self._first_call = True
        
        controls_sequence = np.clip(np.nan_to_num(controls_sequence), self.control_lb, self.control_ub)
        
        opt_bounds = [(lb, ub) for lb, ub in zip(np.tile(self.control_lb, horizon), 
                                                np.tile(self.control_ub, horizon))]
        
        result = minimize(
            self._objective,
            controls_sequence.flatten(),
            args=(current_states, references),
            method='L-BFGS-B',
            jac=True,
            bounds=opt_bounds,
            options={
                'maxiter': 200,  # Increased for better convergence
                'ftol': 1e-6,    # Relaxed slightly
                'eps': 1e-8,
                'disp': False
            }
        )
        
        if not result.success:
            print(f"Optimization failed: {result.message}")
        self.prev_solution = np.nan_to_num(result.x.reshape(horizon, 3))
        return self.prev_solution[0], result.status, result.fun

    def _objective(self, u_flat, current_states, references):
        """Compute the objective function and its gradient."""
        horizon = self.params['horizon']
        u = np.asarray(u_flat).reshape(horizon, 3).astype(np.float32)
        state = torch.tensor(current_states, dtype=torch.float32, device=self.device)
        total_cost = 0.0
        grad = np.zeros_like(u, dtype=np.float32)
        
        W = torch.tensor(self.params['W'], device=self.device, dtype=torch.float32)
        R = torch.tensor(self.params['R'], device=self.device, dtype=torch.float32)
        S = torch.tensor(self.params['S'], device=self.device, dtype=torch.float32)
        
        for t in range(horizon):
            u_t = torch.tensor(u[t], dtype=torch.float32, device=self.device, requires_grad=True)
            nn_input = torch.cat([state, u_t])
            pred = self.model(nn_input.unsqueeze(0)).squeeze()
            if torch.any(torch.isnan(pred)):
                print(f"NaN detected in prediction at timestep {t}")
                return np.inf, grad.flatten()
            
            tracking_error = pred - torch.tensor(references, device=self.device)
            self.integral_error += tracking_error.detach().cpu().numpy() * self.params['dt']
            # Anti-windup: Clip integral error
            self.integral_error = np.clip(self.integral_error, -10.0, 10.0)
            
            state_viol_low = torch.relu(torch.tensor(self.state_bounds[:, 0], device=self.device) - pred)
            state_viol_high = torch.relu(pred - torch.tensor(self.state_bounds[:, 1], device=self.device))
            state_viol = state_viol_low + state_viol_high
            
            cost_track = float(tracking_error @ W @ tracking_error)
            cost_integral = self.params['lambda_integral'] * float(torch.tensor(self.integral_error, 
                                                                               device=self.device) @ W @ 
                                                                  torch.tensor(self.integral_error, 
                                                                               device=self.device))
            cost_con = float(state_viol @ S @ state_viol)
            cost_control = float(u_t @ R @ u_t)
            cost_terminal = self.params['lambda_terminal'] * float(tracking_error @ tracking_error) if t == horizon - 1 else 0
            cost_smooth = 0.0
            if t > 0:
                control_diff = u_t - torch.tensor(u[t-1], device=self.device)
                cost_smooth = 0.5 * float(control_diff @ control_diff)
            
            step_cost = (self.params['lambda_tracking'] * cost_track +
                        cost_integral +
                        self.params['lambda_con'] * cost_con +
                        cost_control +
                        cost_terminal +
                        cost_smooth)
            total_cost += step_cost
            
            step_cost_torch = (self.params['lambda_tracking'] * (tracking_error @ W @ tracking_error) +
                              self.params['lambda_integral'] * (torch.tensor(self.integral_error, 
                                                                            device=self.device) @ W @ 
                                                               torch.tensor(self.integral_error, 
                                                                            device=self.device)) +
                              self.params['lambda_con'] * (state_viol @ S @ state_viol) +
                              u_t @ R @ u_t)
            if t == horizon - 1:
                step_cost_torch += self.params['lambda_terminal'] * (tracking_error @ tracking_error)
            if t > 0:
                step_cost_torch += 0.5 * ((u_t - torch.tensor(u[t-1], device=self.device)) @ 
                                         (u_t - torch.tensor(u[t-1], device=self.device)))
            
            step_cost_torch.backward()
            grad[t] = u_t.grad.detach().cpu().numpy() if u_t.grad is not None else np.zeros(3)
            u_t.grad = None
            
            state = pred.detach()
        
        if np.isnan(total_cost):
            print("NaN detected in total cost")
            return np.inf, grad.flatten()
        
        return total_cost, grad.flatten().astype(np.float64)

def cost_fun_mimo(current_states, prev_controls, references, bounds, model,
                  W, R, lambda_tracking, lambda_terminal, lambda_integral,
                  w_state_con, w_control_con, s, horizon, dt, max_iter, solver=None):
    """Unified MPC cost function."""
    W = np.asarray(W, dtype=np.float32) if not isinstance(W, np.ndarray) else W
    R = np.asarray(R, dtype=np.float32) if not isinstance(R, np.ndarray) else R
    S = np.asarray(s, dtype=np.float32) if not isinstance(s, np.ndarray) else s
    
    if W.ndim == 0:
        W = np.eye(3, dtype=np.float32) * W
    if R.ndim == 0:
        R = np.eye(3, dtype=np.float32) * R
    if S.ndim == 0:
        S = np.eye(3, dtype=np.float32) * S

    params = {
        'W': W,
        'R': R,
        'S': S,
        'lambda_tracking': np.float32(lambda_tracking),
        'lambda_terminal': np.float32(lambda_terminal),
        'lambda_integral': np.float32(lambda_integral),
        'lambda_con': np.float32(w_state_con),
        'horizon': horizon,
        'dt': dt,
        'max_iter': max_iter
    }
    
    if solver is None:
        solver = UnifiedMPCSolver(model, bounds, params)
    
    control, status, cost = solver.solve(
        np.nan_to_num(np.asarray(current_states, dtype=np.float32)),
        np.nan_to_num(np.asarray(prev_controls, dtype=np.float32)),
        np.nan_to_num(np.asarray(references, dtype=np.float32))
    )
    return control, status, cost
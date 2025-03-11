import torch
import numpy as np
from scipy.optimize import minimize

class EnhancedMPCSolver:
    def __init__(self, model, bounds, params):
        """Initialize the MPC solver with the PINN model, bounds, and parameters."""
        self.model = model.to(torch.float32)  # Ensure model uses float32
        self.device = next(model.parameters()).device
        self.bounds = bounds
        self.params = params
        self.prev_solution = None
        
        # Control bounds as numpy arrays
        self.control_lb = np.array([bounds['control'][k][0] for k in ['fa_dot', 'fw_dot', 'u3']], 
                                 dtype=np.float32)
        self.control_ub = np.array([bounds['control'][k][1] for k in ['fa_dot', 'fw_dot', 'u3']], 
                                 dtype=np.float32)
        
        # State bounds as numpy array [min, max] for [Tt, wt, Ts]
        self.state_bounds = np.array([
            [bounds['state']['Tt'][0], bounds['state']['Tt'][1]],
            [bounds['state']['wt'][0], bounds['state']['wt'][1]],
            [bounds['state']['Ts'][0], bounds['state']['Ts'][1]]
        ], dtype=np.float32)

    def solve(self, current_states, prev_controls, references):
        """Solve the MPC optimization problem."""
        horizon = self.params['horizon']
        
        # Warm-start with previous solution or previous controls
        if self.prev_solution is not None and not np.any(np.isnan(self.prev_solution)):
            init_controls = 0.9 * np.roll(self.prev_solution, -1, axis=0) + \
                           0.1 * np.tile(prev_controls, (horizon, 1))
        else:
            init_controls = np.tile(prev_controls, (horizon, 1))
        
        # Ensure initial controls are within bounds and free of NaN
        init_controls = np.clip(np.nan_to_num(init_controls), self.control_lb, self.control_ub)
        
        # Define optimization bounds
        opt_bounds = [(lb, ub) for lb, ub in zip(np.tile(self.control_lb, horizon), 
                                                np.tile(self.control_ub, horizon))]
        
        # Perform optimization using SLSQP
        result = minimize(
            self._objective,
            init_controls.flatten(),
            args=(current_states, references),
            method='L-BFGS-B',
            jac=True,  # Jacobian is provided by _objective
            bounds=opt_bounds,
            options={
                'maxiter': self.params['max_iter'],
                'ftol': 1e-8,  # Tight tolerance for convergence
                'eps': 1e-8,   # Step size for finite differences
                'disp': False  # Suppress optimization messages
            }
        )
        
        # Check optimization success
        if not result.success:
            print(f"Optimization failed: {result.message}")
        
        # Store solution and handle NaN
        self.prev_solution = np.nan_to_num(result.x.reshape(horizon, 3))
        return self.prev_solution[0], result.status, result.fun

    def _objective(self, u_flat, current_states, references):
        """Compute the objective function and its gradient."""
        horizon = self.params['horizon']
        u = u_flat.reshape(horizon, 3).astype(np.float32)
        state = torch.tensor(current_states, dtype=torch.float32, device=self.device)
        total_cost = 0.0
        grad = np.zeros_like(u, dtype=np.float32)
        
        # Weight matrices as tensors
        W = torch.tensor(self.params['W'], device=self.device)  # State tracking weights
        R = torch.tensor(self.params['R'], device=self.device)  # Control effort weights
        S = torch.tensor(self.params['S'], device=self.device)  # State constraint weights
        
        for t in range(horizon):
            u_t = torch.tensor(u[t], dtype=torch.float32, device=self.device, requires_grad=True)
            nn_input = torch.cat([state, u_t])  # [Tt, wt, Ts, fa_dot, fw_dot, u3]
            
            # Model prediction
            pred = self.model(nn_input.unsqueeze(0)).squeeze()  # [Tt, wt, Ts]
            if torch.any(torch.isnan(pred)):
                print(f"NaN detected in prediction at timestep {t}")
                return np.inf, grad.flatten()
            
            # Cost terms
            tracking_error = pred - torch.tensor(references, device=self.device)
            state_viol_low = torch.relu(torch.tensor(self.state_bounds[:, 0], device=self.device) - pred)
            state_viol_high = torch.relu(pred - torch.tensor(self.state_bounds[:, 1], device=self.device))
            state_viol = state_viol_low + state_viol_high
            
            # Individual cost components
            cost_track = float(tracking_error @ W @ tracking_error)
            cost_con = float(state_viol @ S @ state_viol)
            cost_control = float(u_t @ R @ u_t)
            cost_smooth = 0.0
            if t > 0:
                u_prev = torch.tensor(u[t-1], dtype=torch.float32, device=self.device)
                control_diff = u_t - u_prev
                cost_smooth = 5.0 * float((control_diff @ control_diff).item())  # Scaled by 10 * 0.5 = 5
            
            # Total step cost
            step_cost = (self.params['lambda_tracking'] * cost_track +
                         self.params['lambda_con'] * cost_con +
                         cost_control + cost_smooth)
            total_cost += step_cost
            
            # Compute gradient
            step_cost_torch = (self.params['lambda_tracking'] * (tracking_error @ W @ tracking_error) +
                               self.params['lambda_con'] * (state_viol @ S @ state_viol) +
                               u_t @ R @ u_t)
            if t > 0:
                step_cost_torch += 5.0 * (control_diff @ control_diff)  # Scaled smoothing term
            
            step_cost_torch.backward()
            grad[t] = u_t.grad.detach().cpu().numpy() if u_t.grad is not None else np.zeros(3)
            u_t.grad = None  # Clear gradient
            
            # Update state for next iteration
            state = pred.detach()
        
        if np.isnan(total_cost):
            print("NaN detected in total cost")
            return np.inf, grad.flatten()
        
        return total_cost, grad.flatten()

def cost_fun_mimo(current_states, prev_controls, references, bounds, model,
                  W, R, lambda_tracking, lambda_terminal, lambda_integral,
                  w_state_con, w_control_con, s, horizon, dt, max_iter):
    """
    MPC cost function compatible with main_mpc.py.
    """
    # Use passed W and R; define S internally
    S = np.diag([1e6, 1e6, 2e6]).astype(np.float32)  # Strong penalties for state violations
    
    params = {
        'W': W,
        'R': R,
        'S': S,
        'lambda_tracking': lambda_tracking,  # Use passed value
        'lambda_con': np.float32(30.0),      # Strong constraint enforcement
        'horizon': horizon,         # Limit horizon for efficiency
        'dt': dt,
        'max_iter': max_iter
    }
    
    solver = EnhancedMPCSolver(model, bounds, params)
    control, status, cost = solver.solve(
        np.nan_to_num(current_states.astype(np.float32)),
        np.nan_to_num(prev_controls.astype(np.float32)),
        np.nan_to_num(references.astype(np.float32))
    )
    return control, status, cost
import torch
import numpy as np
from scipy.optimize import minimize

def cost_fun_mimo(current_states, prev_controls, references, bounds, model,
                  W, R, lambda_tracking, lambda_terminal, lambda_integral,
                  w_state_con, w_control_con, s, horizon, max_iter, dt):
    
    device = next(model.parameters()).device
    
    # Extract bounds
    control_lb = np.array([bounds['control'][k][0] for k in ['fa_dot', 'fw_dot', 'u3']], dtype=np.float32)
    control_ub = np.array([bounds['control'][k][1] for k in ['fa_dot', 'fw_dot', 'u3']], dtype=np.float32)
    state_bounds = np.array([[bounds['state'][k][0], bounds['state'][k][1]] for k in ['Tt', 'wt', 'Ts']], dtype=np.float32)

    # Optimization function
    def objective(u_flat):
        u = u_flat.reshape(horizon, 3).astype(np.float32)
        state = torch.tensor(current_states, dtype=torch.float32, device=device)
        total_cost = np.float32(0.0)
        grad = np.zeros_like(u, dtype=np.float32)
        
        for t in range(horizon):
            # Create NN input tensor
            control_t = torch.tensor(u[t], dtype=torch.float32, device=device)
            nn_input = torch.cat([state, control_t])
            nn_input.requires_grad_(True)
            
            # Forward pass
            pred = model(nn_input.unsqueeze(0)).squeeze()
            
            # Calculate gradients
            jac = torch.autograd.functional.jacobian(
                lambda x: model(x.unsqueeze(0)), 
                nn_input
            )[:, 3:].detach().cpu().numpy().astype(np.float32)
            
            # Convert prediction
            pred_np = pred.detach().cpu().numpy().astype(np.float32)
            
            # Tracking error
            tracking_error = pred_np - references
            
            # State constraint violations
            state_viol = np.maximum(state_bounds[:, 0] - pred_np, 0) + \
                        np.maximum(pred_np - state_bounds[:, 1], 0)
            
            # Cost components
            cost_tracking = lambda_tracking * np.dot(tracking_error, W @ tracking_error)
            cost_control = np.dot(u[t], R @ u[t])
            cost_state_con = w_state_con * np.dot(state_viol, state_viol)
            
            # Gradient components
            grad_tracking = 2 * lambda_tracking * (jac.T @ W @ tracking_error)
            grad_control = 2 * (R @ u[t])
            grad_state_con = 2 * w_state_con * (jac.T @ state_viol)
            
            total_cost += cost_tracking + cost_control + cost_state_con
            grad[t] = grad_tracking + grad_control + grad_state_con
            
            # Update state
            state = torch.tensor(pred_np, dtype=torch.float32, device=device)
        
        return total_cost.astype(np.float64), grad.flatten().astype(np.float64)

    # Solve optimization
    res = minimize(objective,
                   np.tile(prev_controls.astype(np.float32), horizon),
                   method='L-BFGS-B',
                   jac=True,
                   bounds=[(lb, ub) for lb, ub in zip(np.tile(control_lb, horizon), np.tile(control_ub, horizon))],
                   options={'maxiter': max_iter, 'disp': False})
    
    return res.x[:3].astype(np.float32), None, 0
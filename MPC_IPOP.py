import torch
import numpy as np
import cyipopt  # Install with: pip install cyipopt

class EnhancedMPCSolver:
    def __init__(self, model, bounds, params):
        """Initialize the MPC solver with IPOPT"""
        self.model = model.to(torch.float32)
        self.device = next(model.parameters()).device
        self.bounds = bounds
        self.params = params
        self.prev_solution = None

        # Control bounds
        self.control_lb = np.array([bounds['control'][k][0] for k in ['fa_dot', 'fw_dot', 'u3']], 
                                 dtype=np.float64)  # IPOPT works better with float64
        self.control_ub = np.array([bounds['control'][k][1] for k in ['fa_dot', 'fw_dot', 'u3']], 
                                 dtype=np.float64)

        # State bounds
        self.state_bounds = np.array([
            [bounds['state']['Tt'][0], bounds['state']['Tt'][1]],
            [bounds['state']['wt'][0], bounds['state']['wt'][1]],
            [bounds['state']['Ts'][0], bounds['state']['Ts'][1]]
        ], dtype=np.float64)

    def solve(self, current_states, prev_controls, references):
        """Solve using IPOPT"""
        horizon = self.params['horizon']
        n_controls = 3
        
        # Warm-start initialization
        if self.prev_solution is not None and not np.any(np.isnan(self.prev_solution)):
            init_controls = 0.9 * np.roll(self.prev_solution, -1, axis=0) + \
                           0.1 * np.tile(prev_controls, (horizon, 1))
        else:
            init_controls = np.tile(prev_controls, (horizon, 1))
        
        init_controls = np.clip(np.nan_to_num(init_controls), self.control_lb, self.control_ub)
        x0 = init_controls.astype(np.float64).flatten()

        # IPOPT problem setup
        nvar = horizon * n_controls
        lb = np.tile(self.control_lb, horizon).astype(np.float64)
        ub = np.tile(self.control_ub, horizon).astype(np.float64)

        # Create IPOPT problem
        problem = cyipopt.Problem(
            nvar=nvar,
            x0=x0,
            lb=lb,
            ub=ub,
            cl=np.array([]),  # No constraints besides bounds
            cu=np.array([]),
            m=0,
            problem_obj=IpoptInterface(
                self.model,
                current_states.astype(np.float64),
                references.astype(np.float64),
                self.params,
                self.state_bounds,
                self.device
            )
        )

        # Set IPOPT options
        problem.add_option('max_iter', self.params['max_iter'])
        problem.add_option('tol', 1e-6)
        problem.add_option('print_level', 0)
        problem.add_option('mu_init', 1e-6)
        problem.add_option('hessian_approximation', 'limited-memory')

        # Solve optimization
        x_opt, info = problem.solve(x0)
        
        # Process results
        if info['status'] not in [0, 1]:  # Acceptable status codes
            print(f"IPOPT failed: {info['status_msg'].decode()}")
        
        self.prev_solution = np.nan_to_num(x_opt.reshape(horizon, 3))
        return self.prev_solution[0].astype(np.float32), info['status'], info['obj_val']

class IpoptInterface:
    """Bridge between IPOPT and PyTorch model"""
    def __init__(self, model, current_states, references, params, state_bounds, device):
        self.model = model
        self.current_states = torch.tensor(current_states, dtype=torch.float32, device=device)
        self.references = torch.tensor(references, dtype=torch.float32, device=device)
        self.params = params
        self.state_bounds = state_bounds
        self.device = device
        self.horizon = params['horizon']

    def objective(self, u_flat):
        """IPOPT-compatible objective function"""
        u = u_flat.reshape(self.horizon, 3).astype(np.float32)
        state = self.current_states.clone()
        total_cost = 0.0
        
        W = torch.tensor(self.params['W'], device=self.device)
        R = torch.tensor(self.params['R'], device=self.device)
        S = torch.tensor(self.params['S'], device=self.device)
        
        for t in range(self.horizon):
            u_t = torch.tensor(u[t], dtype=torch.float32, device=self.device, requires_grad=True)
            nn_input = torch.cat([state, u_t])
            
            # Model prediction
            pred = self.model(nn_input.unsqueeze(0)).squeeze()
            
            # Cost calculation
            tracking_error = pred - self.references
            state_viol_low = torch.relu(torch.tensor(self.state_bounds[:, 0], device=self.device) - pred)
            state_viol_high = torch.relu(pred - torch.tensor(self.state_bounds[:, 1], device=self.device))
            state_viol = state_viol_low + state_viol_high
            
            cost = (self.params['lambda_tracking'] * (tracking_error @ W @ tracking_error) +
                    self.params['lambda_con'] * (state_viol @ S @ state_viol) +
                    u_t @ R @ u_t)
            
            if t > 0:
                u_prev = torch.tensor(u[t-1], dtype=torch.float32, device=self.device)
                cost += 5.0 * torch.sum((u_t - u_prev)**2)
            
            total_cost += cost.item()
            state = pred.detach()
        
        return total_cost

    def gradient(self, u_flat):
        """IPOPT-compatible gradient calculation"""
        u = u_flat.reshape(self.horizon, 3).astype(np.float32)
        state = self.current_states.clone()
        grad = np.zeros_like(u, dtype=np.float64)
        
        W = torch.tensor(self.params['W'], device=self.device)
        R = torch.tensor(self.params['R'], device=self.device)
        S = torch.tensor(self.params['S'], device=self.device)
        
        for t in range(self.horizon):
            u_t = torch.tensor(u[t], dtype=torch.float32, device=self.device, requires_grad=True)
            nn_input = torch.cat([state, u_t])
            
            # Model prediction
            pred = self.model(nn_input.unsqueeze(0)).squeeze()
            
            # Cost calculation
            tracking_error = pred - self.references
            state_viol_low = torch.relu(torch.tensor(self.state_bounds[:, 0], device=self.device) - pred)
            state_viol_high = torch.relu(pred - torch.tensor(self.state_bounds[:, 1], device=self.device))
            state_viol = state_viol_low + state_viol_high
            
            cost = (self.params['lambda_tracking'] * (tracking_error @ W @ tracking_error) +
                    self.params['lambda_con'] * (state_viol @ S @ state_viol) +
                    u_t @ R @ u_t)
            
            if t > 0:
                u_prev = torch.tensor(u[t-1], dtype=torch.float32, device=self.device)
                cost += 5.0 * torch.sum((u_t - u_prev)**2)
            
            cost.backward()
            grad[t] = u_t.grad.detach().cpu().numpy().astype(np.float64)
            u_t.grad = None
            state = pred.detach()
        
        return grad.flatten()

# Keep the cost_fun_mimo function identical to your original version
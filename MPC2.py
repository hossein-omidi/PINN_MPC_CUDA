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
        
        # Control bounds with buffer
        self.control_lb = np.array([bounds['control'][k][0] for k in ['fa_dot','fw_dot','u3']], 
                                 dtype=np.float32) + 1e-3
        self.control_ub = np.array([bounds['control'][k][1] for k in ['fa_dot','fw_dot','u3']], 
                                 dtype=np.float32) - 1e-3
        self.state_bounds = np.array([[bounds['state'][k][0], bounds['state'][k][1]] 
                                   for k in ['Tt','wt','Ts']], dtype=np.float32)

        # State constraint penalty matrix
        self.S = np.diag([1e4, 1e4, 1e2]).astype(np.float32)  # Penalize Tt/wt violations heavily

    def solve(self, current_states, prev_controls, references):
        # Enhanced warm-start with reference-aware initialization
        if self.prev_solution is not None:
            controls_sequence = 0.5 * np.roll(self.prev_solution, -3, axis=0) + \
                               0.3 * np.tile(prev_controls, (self.params['horizon'], 1)) + \
                               0.2 * (references[0] - current_states[0]) * np.random.normal(0,0.1,size=(self.params['horizon'],3))
        else:
            controls_sequence = np.tile(prev_controls, (self.params['horizon'], 1))
        
        # Bounds with 1% safety margin
        safety_margin = 0.01*(self.control_ub - self.control_lb)
        bounded_controls = np.clip(controls_sequence, 
                                 self.control_lb + safety_margin, 
                                 self.control_ub - safety_margin)
        
        # Optimization with enhanced settings
        result = minimize(
            self._objective,
            bounded_controls.flatten(),
            args=(current_states, references),
            method='L-BFGS-B',
            jac=True,
            bounds=[(lb, ub) for lb, ub in zip(
                np.tile(self.control_lb, self.params['horizon']),
                np.tile(self.control_ub, self.params['horizon'])
            )],
            options={
                'maxiter': self.params['max_iter'],
                'ftol': 1e-8,
                'gtol': 1e-6,
                'eps': 1e-4
            }
        )
        
        # Solution polishing
        self.prev_solution = self._smooth_solution(result.x.reshape(self.params['horizon'], 3))
        return self.prev_solution[0], None, 0

    def _smooth_solution(self, solution):
        # Apply moving average filter to control sequence
        window_size = min(5, self.params['horizon']//4)
        if window_size > 1:
            return np.convolve(solution, np.ones(window_size)/window_size, mode='same')
        return solution

    def _objective(self, u_flat, current_states, references):
        device = next(self.model.parameters()).device
        u = u_flat.reshape(self.params['horizon'], 3).astype(np.float32)
        state = torch.tensor(current_states, dtype=torch.float32, device=device)
        total_cost = 0.0
        grad = np.zeros_like(u)
        
        for t in range(self.params['horizon']):
            control_tensor = torch.tensor(u[t], device=device, dtype=torch.float32)
            nn_input = torch.cat([state, control_tensor])
            nn_input.requires_grad = True
            pred = self.model(nn_input.unsqueeze(0)).squeeze()
            
            # State predictions and constraints
            next_state = pred.detach().cpu().numpy().astype(np.float32)
            state_violation = np.maximum(next_state - self.state_bounds[:,1], 
                                       self.state_bounds[:,0] - next_state, 
                                       np.zeros_like(next_state))
            
            # Cost components
            tracking_error = next_state - references
            self.integral_error += tracking_error * self.params['dt']
            
            # Enhanced cost function with:
            # - Dynamic tracking weights
            # - State constraint penalties
            # - Control smoothness term
            W_dynamic = self.params['W'] * np.exp(-0.1*np.linalg.norm(tracking_error))
            cost_track = np.dot(tracking_error, W_dynamic @ tracking_error)
            cost_state_con = np.dot(state_violation, self.S @ state_violation)
            cost_control = np.dot(u[t], self.params['R'] @ u[t])
            cost_smooth = 0.1*np.linalg.norm(u[t] - u[t-1]) if t > 0 else 0
            
            total_cost += (self.params['lambda_tracking'] * cost_track +
                         cost_state_con +
                         self.params['lambda_control'] * cost_control +
                         cost_smooth)

            # Gradient calculations with constraint awareness
            jac = torch.autograd.functional.jacobian(
                lambda x: self.model(x.unsqueeze(0)), 
                nn_input
            )[0, :, 3:].detach().cpu().numpy().astype(np.float32)
            
            grad_track = 2 * self.params['lambda_tracking'] * (jac.T @ (W_dynamic @ tracking_error))
            grad_state_con = 2 * (jac.T @ (self.S @ state_violation))
            grad_control = 2 * self.params['lambda_control'] * (self.params['R'] @ u[t])
            grad_smooth = 0.2*(u[t] - u[t-1]) if t > 0 else 0
            
            grad[t] = grad_track + grad_state_con + grad_control + grad_smooth
            
            state = torch.tensor(next_state, dtype=torch.float32, device=device)
        
        return total_cost.astype(np.float64), grad.flatten().astype(np.float64)

def cost_fun_mimo(current_states, prev_controls, references, bounds, model,
                  W, R, lambda_tracking, lambda_terminal, lambda_integral,
                  w_state_con, w_control_con, s, horizon, dt, max_iter):
    
    params = {
        'W': W.astype(np.float32),
        'R': R.astype(np.float32),
        'lambda_tracking': np.float32(lambda_tracking),
        'lambda_control': np.float32(0.5),  # New control weight
        'horizon': min(horizon, 30),  # Limit horizon for stability
        'dt': dt,
        'max_iter': max_iter
    }
    
    solver = MPCSolver(model, bounds, params)
    return solver.solve(
        current_states.astype(np.float32),
        prev_controls.astype(np.float32),
        references.astype(np.float32)
    )
import torch
import numpy as np
import casadi as ca

class NeuralNetworkCallback(ca.Callback):
    def __init__(self, model, device, name='nn_callback', opts={}):
        """Callback to evaluate the neural network within CasADi with finite differences."""
        ca.Callback.__init__(self)
        self.model = model
        self.device = device
        # Enable finite differences for derivative approximation
        opts.update({
            'enable_fd': True,           # Enable finite difference computation
            'fd_method': 'central'       # Use central differences for accuracy
        })
        self.construct(name, opts)

    def get_n_in(self):
        """Number of inputs to the callback."""
        return 1  # Single input vector (states + controls)

    def get_n_out(self):
        """Number of outputs from the callback."""
        return 1  # Single output vector (predicted states)

    def get_sparsity_in(self, i):
        """Input sparsity: 6x1 (3 states + 3 controls)."""
        return ca.Sparsity.dense(6, 1)

    def get_sparsity_out(self, i):
        """Output sparsity: 3x1 (predicted states)."""
        return ca.Sparsity.dense(3, 1)

    def eval(self, arg):
        """Evaluate the neural network numerically."""
        nn_input = np.array(arg[0]).astype(np.float32)  # Convert CasADi input to NumPy
        with torch.no_grad():
            nn_input_tensor = torch.tensor(nn_input, dtype=torch.float32, device=self.device)
            # Assuming model expects (1, 6) input and outputs (1, 3)
            pred_tensor = self.model(nn_input_tensor.reshape(1, -1)).squeeze(0)
        return [ca.DM(pred_tensor.cpu().numpy().reshape(3, 1))]  # Return 3x1 DM object

class EnhancedMPCSolver:
    def __init__(self, model, bounds, params):
        """Initialize the MPC solver with CasADi."""
        self.model = model.to(torch.float32)
        self.device = next(model.parameters()).device
        self.bounds = bounds
        self.params = params
        self.prev_solution = None

        # Control bounds
        self.control_lb = np.array([bounds['control'][k][0] for k in ['fa_dot', 'fw_dot', 'u3']],
                                   dtype=np.float64)
        self.control_ub = np.array([bounds['control'][k][1] for k in ['fa_dot', 'fw_dot', 'u3']],
                                   dtype=np.float64)

        # State bounds
        self.state_bounds = np.array([
            [bounds['state']['Tt'][0], bounds['state']['Tt'][1]],
            [bounds['state']['wt'][0], bounds['state']['wt'][1]],
            [bounds['state']['Ts'][0], bounds['state']['Ts'][1]]
        ], dtype=np.float64)

        # Initialize neural network callback
        self.nn_callback = NeuralNetworkCallback(self.model, self.device)
        print("MPC Solver Initialized: Horizon =", self.params['horizon'], 
              "| Controls =", len(self.control_lb), "| States =", self.state_bounds.shape[0])

    def solve(self, current_states, prev_controls, references):
        """Solve the MPC problem using CasADi."""
        horizon = self.params['horizon']
        n_controls = 3

        print("Starting MPC Solve...")
        print(f"Current States: {current_states.flatten()}")
        print(f"Previous Controls: {prev_controls}")
        print(f"References: {references.flatten()}")

        # Warm-start initialization
        if self.prev_solution is not None and not np.any(np.isnan(self.prev_solution)):
            init_controls = 0.9 * np.roll(self.prev_solution, -1, axis=0) + \
                            0.1 * np.tile(prev_controls, (horizon, 1))
        else:
            init_controls = np.tile(prev_controls, (horizon, 1))
        init_controls = np.clip(np.nan_to_num(init_controls), self.control_lb, self.control_ub)
        x0 = init_controls.astype(np.float64).flatten()
        print(f"Initial Control Guess Shape: {init_controls.shape}")

        # CasADi problem setup
        self.opti = ca.Opti()  # Store opti as instance variable for debugging if needed
        u = self.opti.variable(horizon, n_controls)
        self.opti.set_initial(u, x0.reshape(horizon, n_controls))
        print("Optimization Problem Defined: Decision Variables Shape =", u.shape)

        # Parameters
        current_state_param = self.opti.parameter(3, 1)
        reference_param = self.opti.parameter(3, 1)
        self.opti.set_value(current_state_param, current_states.reshape(3, 1).astype(np.float64))
        self.opti.set_value(reference_param, references.reshape(3, 1).astype(np.float64))

        # Objective function
        total_cost = 0
        state = current_state_param

        for t in range(horizon):
            u_t = u[t, :].T  # 3x1

            # Neural network prediction using callback
            nn_input = ca.vertcat(state, u_t)  # 6x1 symbolic
            pred = self.nn_callback(nn_input)  # 3x1 symbolic
            print(f"Horizon Step {t}: Predicted State = {pred}")

            # Tracking error
            tracking_error = pred - reference_param  # 3x1

            # State constraint violations
            state_viol_low = ca.fmax(self.state_bounds[:, 0].reshape(3, 1) - pred, 0)
            state_viol_high = ca.fmax(pred - self.state_bounds[:, 1].reshape(3, 1), 0)
            state_viol = state_viol_low + state_viol_high  # 3x1

            # Cost components
            W = ca.DM(self.params['W'])  # 3x3
            R = ca.DM(self.params['R'])  # 3x3
            S = ca.DM(self.params['S'])  # 3x3

            tracking_cost = self.params['lambda_tracking'] * ca.mtimes([tracking_error.T, W, tracking_error])
            violation_cost = self.params['lambda_con'] * ca.mtimes([state_viol.T, S, state_viol])
            control_cost = ca.mtimes([u_t.T, R, u_t])

            cost = tracking_cost + violation_cost + control_cost

            if t > 0:
                u_prev = u[t-1, :].T
                cost += 5.0 * ca.sumsqr(u_t - u_prev)

            total_cost += cost
            state = pred
            print(f"Horizon Step {t}: Cost Contribution = {cost}")

        self.opti.minimize(total_cost)
        print("Objective Function Set: Total Cost Expression Defined")

        # Control constraints
        lb_matrix = ca.repmat(ca.DM(self.control_lb).T, horizon, 1)  # horizon x 3
        ub_matrix = ca.repmat(ca.DM(self.control_ub).T, horizon, 1)  # horizon x 3
        u_vec = ca.vec(u)  # Flatten to (horizon * n_controls) x 1
        lb_vec = ca.vec(lb_matrix)
        ub_vec = ca.vec(ub_matrix)
        self.opti.subject_to(u_vec >= lb_vec)
        self.opti.subject_to(u_vec <= ub_vec)
        print(f"Constraints Applied: Control Bounds Shape = {lb_matrix.shape}")

        # Solver options
        opts = {
            'ipopt.print_level': 0,              # Suppress output
            'print_time': 0,                     # Suppress timing info
            'ipopt.max_iter': self.params['max_iter'],  # Max iterations
            'ipopt.tol': 1e-6,                   # Convergence tolerance
            'ipopt.hessian_approximation': 'limited-memory',  # Use limited-memory BFGS
            'ipopt.derivative_test': 'first-order',  # Optional: Check derivatives
            'ipopt.derivative_test_perturbation': 1e-8  # Perturbation for finite differences
        }
        self.opti.solver('ipopt', opts)
        print("Solver Configured with IPOPT")

        # Solve
        try:
            sol = self.opti.solve()
            u_opt = sol.value(u)
            status = 0
            cost_val = sol.value(total_cost)
            print(f"Solve Successful: Optimal Control (first step) = {u_opt[0]}, Cost = {cost_val}")
        except Exception as e:
            print(f"CasADi optimization failed: {str(e)}")
            u_opt = self.opti.debug.value(u) if self.opti.debug else np.full((horizon, n_controls), np.nan)
            status = -1
            cost_val = float('inf')
            print("Solve Failed: Using debug or default values")

        self.prev_solution = np.nan_to_num(u_opt)
        print(f"Solution Stored: Status = {status}")
        return self.prev_solution[0].astype(np.float32), status, cost_val

def cost_fun_mimo(current_states, prev_controls, references, bounds, model,
                  W, R, lambda_tracking, lambda_terminal, lambda_integral,
                  w_state_con, w_control_con, s, horizon, dt, max_iter):
    """MPC cost function compatible with the main simulation."""
    S = np.diag([1e6, 1e6, 2e6]).astype(np.float32)

    params = {
        'W': W.astype(np.float32),
        'R': R.astype(np.float32),
        'S': S,
        'lambda_tracking': float(lambda_tracking),
        'lambda_con': np.float32(30.0),
        'horizon': min(horizon, 15),
        'dt': float(dt),
        'max_iter': int(max_iter)
    }

    solver = EnhancedMPCSolver(model, bounds, params)
    control, status, cost = solver.solve(
        np.nan_to_num(current_states.astype(np.float32)),
        np.nan_to_num(prev_controls.astype(np.float32)),
        np.nan_to_num(references.astype(np.float32))
    )
    return control, status, cost
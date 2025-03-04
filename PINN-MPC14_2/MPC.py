import torch
import numpy as np


def cost_fun_mimo(current_states, prev_controls, references, bounds, model,
                  W=np.diag([1000, 1000, 100]),  # State tracking weights (Tt, wt, Ts)
                  R=np.diag([0.01, 0.01, 0.1]),  # Control effort weights
                  lambda_tracking=100,  # Tracking weight
                  lambda_terminal=0.1,  # Terminal cost weight
                  lambda_integral=50,  # Integral cost weight
                  lambda_smooth=0.01,  # Control smoothness weight
                  w_state_con=1e6,  # State constraint weight
                  w_control_con=1e6,  # Control constraint weight
                  s=1e-4,  # Step size
                  horizon=10,  # Prediction horizon steps
                  dt=0.05,  # MPC update interval
                  max_iter=200):  # Optimization iterations

    device = next(model.parameters()).device
    control_names = ['fa_dot', 'fw_dot', 'u3']
    state_names = ['Tt', 'wt', 'Ts']

    # Initialize control sequence
    controls_sequence = np.tile(prev_controls, (horizon, 1))
    best_controls = controls_sequence.copy()
    best_cost = float('inf')

    # Integral error for all states
    if not hasattr(cost_fun_mimo, "integral_error"):
        cost_fun_mimo.integral_error = np.zeros(3)  # [Tt, wt, Ts]

    # Extract bounds
    control_lb = np.array([bounds['control'][k][0] for k in control_names])
    control_ub = np.array([bounds['control'][k][1] for k in control_names])
    state_lb = np.array([bounds['state'][k][0] for k in state_names])
    state_ub = np.array([bounds['state'][k][1] for k in state_names])

    for _ in range(max_iter):
        total_cost = 0
        grad_total = np.zeros((horizon, 3))  # Gradient for all controls
        hessian_total = np.zeros((horizon * 3, horizon * 3))  # Hessian matrix

        # State sequence prediction
        state_sequence = [current_states.copy()]
        for t in range(horizon):
            # Predict next state
            nn_input = np.concatenate([state_sequence[-1], controls_sequence[t]])
            x = torch.FloatTensor(nn_input).to(device).unsqueeze(0)
            x.requires_grad = True
            y_pred = model(x)
            next_state = y_pred.detach().cpu().numpy().flatten()
            state_sequence.append(next_state)

            # Compute Jacobian [3x3] for all states and controls
            jac = []
            for i in range(3):
                grad_output = torch.zeros_like(y_pred)
                grad_output[:, i] = 1
                jac_i = torch.autograd.grad(y_pred, x, grad_outputs=grad_output,
                                            retain_graph=True, create_graph=False)[0][:, 3:]
                jac.append(jac_i.squeeze(0).cpu().numpy())
            jac = np.array(jac)  # [3, 3]

            # Calculate errors for all states
            tracking_error = next_state - references  # [Tt, wt, Ts]
            cost_fun_mimo.integral_error += tracking_error * dt  # Update integral error

            # Cost components
            # Tracking cost with weight matrix W
            weighted_error = W @ tracking_error
            integral_cost = lambda_integral * (W @ cost_fun_mimo.integral_error)
            tracking_cost = weighted_error @ weighted_error + integral_cost @ integral_cost

            # Control effort cost
            control_cost = controls_sequence[t] @ R @ controls_sequence[t]

            # Constraint violation costs
            state_viol = np.maximum(state_lb - next_state, 0) + np.maximum(next_state - state_ub, 0)
            control_viol = ((controls_sequence[t] < control_lb) | (controls_sequence[t] > control_ub)).astype(float)
            constraint_cost = w_state_con * (state_viol @ state_viol) + w_control_con * (control_viol @ control_viol)

            # Gradient components
            grad_tracking = 2 * jac.T @ W @ (tracking_error + lambda_integral * cost_fun_mimo.integral_error)
            grad_control = 2 * R @ controls_sequence[t]
            grad_smooth = 2 * lambda_smooth * (controls_sequence[t] -
                                               (controls_sequence[t - 1] if t > 0 else prev_controls))
            grad_constraints = 2 * (w_state_con * state_viol + w_control_con * control_viol)

            total_grad = grad_tracking + grad_control + grad_smooth + grad_constraints
            grad_total[t] = total_grad

            # Hessian components
            hess_tracking = 2 * jac.T @ W @ jac
            hess_control = 2 * R
            hess_smooth = 2 * lambda_smooth * np.eye(3)
            hess_constraints = 2 * np.diag([w_state_con] * 3 + [w_control_con] * 3)[:3, :3]

            hessian_total[t * 3:(t + 1) * 3, t * 3:(t + 1) * 3] = (hess_tracking + hess_control +
                                                                   hess_smooth + hess_constraints)

            # Total cost accumulation
            total_cost += tracking_cost + control_cost + constraint_cost + lambda_terminal * (
                        next_state @ W @ next_state)

        # Optimization step
        try:
            du = -np.linalg.solve(hessian_total, grad_total.flatten())
        except np.linalg.LinAlgError:
            du = -np.linalg.pinv(hessian_total) @ grad_total.flatten()

        # Update controls with step size
        controls_sequence = controls_sequence + s * du.reshape(horizon, 3)
        controls_sequence = np.clip(controls_sequence, control_lb, control_ub)

        # Check for improvement
        if total_cost < best_cost:
            best_cost = total_cost
            best_controls = controls_sequence.copy()

    # Return first control input and predicted state
    return best_controls[0], state_sequence[-1], 0
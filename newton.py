import numpy as np


def line_search(x, delta_x, f, grad_f, alpha=0.1, beta=0.5, max_it=25):
    """
    Backtracking line search.

    Parameters
    ----------
    x : array_like, shape (n,)
        Current point.
    delta_x : array_like, shape (n,)
        Descent direction.
    f : callable
        Objective function; f(x) -> scalar.
    grad_f : callable
        Gradient; grad_f(x) -> array_like shape (n,).
    alpha : float,
        Sufficient decrease parameter (0 < alpha < 0.5).
    beta : float,
        Shrinkage factor (0 < beta < 1).
    max_it : int,
        Maximum number of backtracking steps.

    Returns
    -------
    t : float
        Step size
    """
    t = 1

    # Precompute some constants
    fx = f(x)
    grad_fx_delta_x = np.dot(grad_f(x), delta_x)

    # Check delta_x is a descent direction
    if grad_fx_delta_x >= 0:
        raise RuntimeError("delta_x isn't a descent direction!")

    # Backtracking loop
    for it in range(max_it):
        if f(x + t * delta_x) <= fx + alpha * t * grad_fx_delta_x:
            break
        t *= beta
    else:
        print("Warning: maximum line search iteration limit reached")

    return t


def newton(x_0, f, grad_f, hess_f, eps=1e-3, alpha=0.1, beta=0.5, max_it=100, A=None, b=None):
    """
    Newton's method for minimisation with backtracking line search.
    Handles unconstrained and equality-constrained problems Ax = b.

    Parameters
    ----------
    x_0 : array_like, shape (n,)
        Initial point for the iteration.
    f : callable
        Objective function; f(x) → float.
    grad_f : callable
        Gradient function; grad_f(x) → ndarray of shape (n,).
    hess_f : callable
        Hessian function; hess_f(x) → ndarray of shape (n, n).
    eps : float, optional
        Stopping tolerance based on the Newton decrement λ(x). The solver stops when λ(x)^2 / 2 ≤ eps.
    alpha : float, optional
        Backtracking line search sufficient decrease parameter (0 < alpha < 0.5).
    beta : float, optional
        Backtracking line search shrinkage factor (0 < beta < 1).
    max_it : int, optional
        Maximum number of Newton iterations.
    A : ndarray, shape (p, n), optional
        Equality constraint matrix (Ax = b).
    b : ndarray, shape (p,), optional
        Equality constraint vector (Ax = b).

    Returns
    -------
    x : ndarray, shape (n,)
        The estimated minimiser of f.
    it : int
        Number of Newton iterations performed.
    path : list
        list of all iterates x_k

    """ 
    x = np.array(x_0.copy(), dtype=float)
    path = [x.copy()]
    n = len(x)

    for it in range(max_it):
        # Gradient and Hessian
        g = grad_f(x)
        H = hess_f(x)

        if A is not None and b is not None:
            # KKT system for equality constraints: [H A^T] [dx] = [-g]
            #                                       [A  0 ] [lam]   [0]
            p = A.shape[0]
            KKT = np.block([[H, A.T],
                           [A, np.zeros((p, p))]])
            rhs = np.concatenate([-g, np.zeros(p)])
            
            try:
                sol = np.linalg.solve(KKT, rhs)
                delta_x = sol[:n]
            except np.linalg.LinAlgError:
                print("Warning: Singular KKT matrix")
                break
        else:
            # Unconstrained: Newton step
            try:
                delta_x = -np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                print("Warning: Singular Hessian matrix")
                break

        # Newton decrement
        lmbda_sq = np.dot(delta_x, H @ delta_x)

        if lmbda_sq / 2 <= eps:
            break

        # Line search
        t = line_search(x, delta_x, f, grad_f, alpha, beta)
        x += t * delta_x
        path.append(x.copy())
    else:
        print("Warning: maximum Newton iteration limit reached")

    return x, it, np.array(path)


def newton_barrier_eq(x_0, f, grad_f, hess_f, h, grad_h, hess_h, A=None, b=None,
                       t_0=1.0, mu=10.0, eps=1e-3, alpha=0.1, beta=0.5, 
                       max_outer=20, max_inner=50):
    """
    Newton's method with logarithmic barrier for inequality constraints h(x) <= 0
    and equality constraints Ax = b.

    Parameters
    ----------
    x_0 : array_like, shape (n,)
        Initial feasible point.
    f : callable
        Objective function; f(x) → float.
    grad_f : callable
        Gradient of f; grad_f(x) → ndarray of shape (n,).
    hess_f : callable
        Hessian of f; hess_f(x) → ndarray of shape (n, n).
    h : callable
        Inequality constraint function; h(x) → ndarray of shape (m,).
        Constraints are h_i(x) <= 0.
    grad_h : callable
        Gradient of h; grad_h(x) → ndarray of shape (m, n).
    hess_h : callable
        Hessian of h; hess_h(x) → list of m ndarrays of shape (n, n).
    A : ndarray, shape (p, n), optional
        Equality constraint matrix (Ax = b).
    b : ndarray, shape (p,), optional
        Equality constraint vector (Ax = b).
    t_0 : float, optional
        Initial barrier parameter.
    mu : float, optional
        Barrier parameter increase factor.
    eps : float, optional
        Tolerance for Newton decrement.
    alpha : float, optional
        Backtracking line search parameter.
    beta : float, optional
        Backtracking line search shrinkage factor.
    max_outer : int, optional
        Maximum number of outer (barrier) iterations.
    max_inner : int, optional
        Maximum number of inner (Newton) iterations per barrier iteration.

    Returns
    -------
    x : ndarray, shape (n,)
        The estimated minimiser of f subject to h(x) <= 0 and Ax = b.
    outer_iter : int
        Number of outer iterations completed.
    path : list
        List of all iterates x_k.
    """
    x = np.array(x_0.copy(), dtype=float)
    path = [x.copy()]
    t = t_0

    for outer_it in range(max_outer):
        # Define barrier function and its derivatives
        def phi(x_eval):
            h_vals = h(x_eval)
            if np.any(h_vals >= 0):
                return 1e10
            return f(x_eval) - np.sum(np.log(-h_vals)) / t

        def grad_phi(x_eval):
            h_vals = h(x_eval)
            grad_h_vals = grad_h(x_eval)
            return grad_f(x_eval) + np.sum(grad_h_vals / (-h_vals)[:, np.newaxis], axis=0) / t

        def hess_phi(x_eval):
            h_vals = h(x_eval)
            grad_h_vals = grad_h(x_eval)
            hess_h_vals = hess_h(x_eval)
            H = hess_f(x_eval).copy()
            for i in range(len(h_vals)):
                H += (grad_h_vals[i:i+1].T @ grad_h_vals[i:i+1]) / (-h_vals[i])**2 / t
                H += hess_h_vals[i] / (-h_vals[i]) / t
            return H

        # Solve barrier subproblem using Newton's method
        x, _, inner_path = newton(x, phi, grad_phi, hess_phi, eps=eps, alpha=alpha, beta=beta, 
                                   max_it=max_inner, A=A, b=b)
        
        # Append all inner iterates to path (skip first since it's already in path)
        path.extend(inner_path[1:].tolist())
        
        # Check convergence (primal-dual gap)
        m = len(h(x))
        if m / t <= 1e-6:
            break
        
        t *= mu

    return x, outer_it, np.array(path)


def newton_barrier_phase1(x_0, h, grad_h, hess_h, A=None, b=None,
                          t_0=1.0, mu=10.0, eps=1e-3, alpha=0.1, beta=0.5,
                          max_outer=20, max_inner=50, early_tol=-1e-6):
    """
    Phase I helper: augment x with slack s and minimise s subject to h(x)-s <= 0, Ax=b.

    Returns the first strictly feasible point found (s < early_tol) and the truncated path.
    """
    x0 = np.array(x_0, dtype=float)
    # Initialize slack to make augmented problem strictly feasible
    # Need h(x0) - s0 < 0 for all constraints, so s0 > max(h(x0))
    s0 = np.max(h(x0)) + 1.0  # Add buffer for strict feasibility
    x0_aug = np.concatenate([x0, [s0]]).astype(float)

    # Augmented objective: minimize s (with regularization for numerical stability)
    def f_phase1(xs):
        # Minimize s + 0.5*||x||^2 to ensure strict convexity and keep x bounded
        x_part = xs[:-1]
        s_val = xs[-1]
        return s_val + 0.5 * np.dot(x_part, x_part)

    def grad_f_phase1(xs):
        x_part = xs[:-1]
        g = np.zeros(len(xs))
        g[:-1] = 1.0 * x_part  # Gradient of regularization
        g[-1] = 1.0
        return g

    def hess_f_phase1(xs):
        H = np.zeros((len(xs), len(xs)))
        H[:-1, :-1] = 1.0 * np.eye(len(xs) - 1)  # Hessian of regularization
        return H

    def h_phase1(xs):
        x_part = xs[:-1]
        s_val = xs[-1]
        return h(x_part) - s_val

    def grad_h_phase1(xs):
        x_part = xs[:-1]
        grad_h_x = grad_h(x_part)
        m = grad_h_x.shape[0]
        return np.hstack([grad_h_x, -np.ones((m, 1))])

    def hess_h_phase1(xs):
        x_part = xs[:-1]
        hess_h_x = hess_h(x_part)
        n_x = len(x_part)
        hess_padded = []
        for H in hess_h_x:
            H_pad = np.zeros((n_x+1, n_x+1))
            H_pad[:n_x, :n_x] = H
            hess_padded.append(H_pad)
        return hess_padded

    # Augmented equality matrix (add zero column for s)
    if A is not None:
        A_aug = np.hstack([A, np.zeros((A.shape[0], 1))])
    else:
        A_aug = None

    # Run barrier method on augmented problem, checking for early termination
    x = x0_aug.copy()
    outer_path = [x.copy()]
    t = t_0

    for outer_it in range(max_outer):
        # Define barrier function and its derivatives
        def phi(xs):
            hvals = h_phase1(xs)
            if np.any(hvals >= 0):
                return 1e10
            return f_phase1(xs) - np.sum(np.log(-hvals)) / t

        def grad_phi(xs):
            hvals = h_phase1(xs)
            grad_h_vals = grad_h_phase1(xs)
            return grad_f_phase1(xs) + np.sum(grad_h_vals / (-hvals)[:, np.newaxis], axis=0) / t

        def hess_phi(xs):
            hvals = h_phase1(xs)
            grad_h_vals = grad_h_phase1(xs)
            hess_h_vals = hess_h_phase1(xs)
            H = hess_f_phase1(xs).copy()
            for i in range(len(hvals)):
                H += (grad_h_vals[i:i+1].T @ grad_h_vals[i:i+1]) / (-hvals[i])**2 / t
                H += hess_h_vals[i] / (-hvals[i]) / t
            return H

        # Solve barrier subproblem using Newton's method
        x, _, inner_path = newton(x, phi, grad_phi, hess_phi, eps=eps, alpha=alpha, beta=beta,
                                   max_it=max_inner, A=A_aug, b=b)
        
        # Check for early termination after Newton solve
        if x[-1] < early_tol:
            outer_path.append(x.copy())
            path_arr = np.array(outer_path)
            x_feasible = path_arr[-1, :-1]
            outer_iters_done = path_arr.shape[0] - 1
            return np.array(x_feasible), outer_iters_done, path_arr
        
        outer_path.append(x.copy())
        
        # Check convergence (primal-dual gap)
        m = len(h_phase1(x))
        if m / t <= 1e-6:
            break
        
        t *= mu

    # Finished without early stop: return last x and outer_path
    path_arr = np.array(outer_path)
    outer_iters_done = path_arr.shape[0] - 1
    return np.array(x[:-1]), outer_iters_done, path_arr



if __name__ == "__main__":
    def f_0(x):
        """
        Objective function.
        """
        return np.exp(x[0]) + np.exp(x[1]) + 0.5*(x[0]**2 + x[1]**2)

    def grad_f_0(x):
        """
        Gradient of f_0.
        """
        return np.array([np.exp(x[0]) + x[0],
                        np.exp(x[1]) + x[1]])

    def hess_f_0(x):
        """
        Hessian of f_0.
        """
        return np.array([[np.exp(x[0]) + 1.0, 0.0],
                        [0.0, np.exp(x[1]) + 1.0]])

    # Starting points
    x0 = np.array([-1.5, 2.5])

    # Solve using Newton's method
    x_star, it, path = newton(x0, f_0, grad_f_0, hess_f_0, eps=1e-6)

    print(f"Initial point:      {x0}")
    print(f"Converged in:       {it} iterations")
    print(f"Optimal solution:   {x_star}")
    print(f"Optimal value:      {f_0(x_star)}")



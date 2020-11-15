import numpy as np
import gpytorch
import torch
import torch.nn as nn
from gpytorch.models import ExactGP
from scipy.optimize import fmin_l_bfgs_b

domain = np.array([[0, 5]])

x0_init_guess = domain[:, 0] + (domain[:, 1] - domain[:, 0]) * np.random.rand(domain.shape[0])

""" Solution """


class ExactGP(ExactGP):
    def __init__(self, train_x, train_y, mean_module, likelihood=gpytorch.likelihoods.GaussianLikelihood()):
        super().__init__(train_x, train_y, likelihood=likelihood)
        self.mean_module = mean_module
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=5 / 2))
        self.covar_module.base_kernel.lengthscale = 0.5

    def forward(self, x):
        """Forward computation of GP."""
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    @property
    def output_scale(self):
        """Get output scale."""
        return self.covar_module.outputscale


class GPUCB(nn.Module):
    def __init__(self, gp, beta=2.0):
        super().__init__()
        self.gp = gp
        self.gp.eval()
        self.gp.likelihood.eval()
        self.x = (torch.linspace(-1, 6, 1000) - 2.5) / 2        # TODO: take from GP or x0_init?
        self.beta = beta
        self.update_acquisition_function()

    def update_gp(self, new_inputs, new_targets):
        """Update GP with new points."""
        inputs = torch.cat((self.gp.train_inputs[0], new_inputs.unsqueeze(-1)), dim=0)
        targets = torch.cat((self.gp.train_targets, new_targets), dim=-1)
        self.gp.set_train_data(inputs, targets, strict=False)
        self.update_acquisition_function()

    def get_best_value(self):
        idx = self.gp.train_targets.argmax()
        if len(self.gp.train_targets) == 1:
            xmax, ymax = self.gp.train_inputs[idx], self.gp.train_targets[idx]
        else:
            xmax, ymax = self.gp.train_inputs[0][idx], self.gp.train_targets[idx]
        return xmax, ymax

    def update_acquisition_function(self):
        pred = self.gp(self.x)
        ucb = pred.mean + self.beta * pred.stddev  # Calculate UCB.
        self._acquisition_function = ucb

    @property
    def acquisition_function(self):
        return self._acquisition_function

    def forward(self):
        """Call the algorithm. """
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            y = self.acquisition_function
            max_id = torch.argmax(y)
            next_point = self.x[[[max_id]]]
        return next_point


class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration. """
        # TODO: enter your code here
        self.f_GPUCB = None
        self.v_GPUCB = None

    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: np.ndarray
            1 x domain.shape[0] array containing the next point to evaluate
        """

        # TODO: enter your code here
        if self.f_GPUCB is None and self.v_GPUCB is None:
            return x0_init_guess
        else:
            return self.optimize_acquisition_function()         # return x with highest optimized f

        # In implementing this function, you may use optimize_acquisition_function() defined below.

    def optimize_acquisition_function(self):
        """
        Optimizes the acquisition function.

        Returns
        -------
        x_opt: np.ndarray
            1 x domain.shape[0] array containing the point that maximize the acquisition function.
        """
        self.v_GPUCB.update_acquisition_function()
        self.f_GPUCB.update_acquisition_function()

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick best solution
        for _ in range(20):
            x0 = x0_init_guess
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=domain,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *domain[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        return np.atleast_2d(x_values[ind])

    def acquisition_function(self, x):
        """
        Compute the acquisition function.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f

        Returns
        ------
        af_value: float
            Value of the acquisition function at x
        """

        # TODO: enter your code here
        return self.f_GPUCB()

    def add_data_point(self, x, f, v):
        """
        Add data points to the model.

        Parameters
        ----------
        x: np.ndarray
            Hyperparameters
        f: np.ndarray
            Model accuracy
        v: np.ndarray
            Model training speed
        """
        # TODO: enter your code here
        if self.f_GPUCB is None and self.v_GPUCB is None:
            self.f_GPUCB = GPUCB(ExactGP(x, f, gpytorch.means.ZeroMean()))
            self.v_GPUCB = GPUCB(ExactGP(x, v, gpytorch.means.ConstantMean(1.5)))
        else:
            self.f_GPUCB.update_gp(x, f)
            self.v_GPUCB.update_gp(x, v)

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: np.ndarray
            1 x domain.shape[0] array containing the optimal solution of the problem
        """

        # TODO: enter your code here
        raise NotImplementedError


""" Toy problem to check code works as expected """


def check_in_domain(x):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= domain[None, :, 0]) and np.all(x <= domain[None, :, 1])


def f(x):
    """Dummy objective"""
    mid_point = domain[:, 0] + 0.5 * (domain[:, 1] - domain[:, 0])
    return - np.linalg.norm(x - mid_point, 2)  # -(x - 2.5)^2


def v(x):
    """Dummy speed"""
    return 2.0


def main():
    # Init problem
    agent = BO_algo()

    # Loop until budget is exhausted
    for j in range(20):                         # query_new_point()
        # Get next recommendation
        x = agent.next_recommendation()         # algorithm() aka call GPUCB

        # Check for valid shape
        assert x.shape == (1, domain.shape[0]), \
            f"The function next recommendation must return a numpy array of " \
            f"shape (1, {domain.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x)                          # objective_function
        cost_val = v(x)                         # objective_function
        agent.add_data_point(x, obj_val, cost_val) # regret.append() AND algorithm.update_gp(x, y)

    # Validate solution
    solution = np.atleast_2d(agent.get_solution())
    assert solution.shape == (1, domain.shape[0]), \
        f"The function get solution must return a numpy array of shape (" \
        f"1, {domain.shape[0]})"
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the ' \
        f'domain, {solution} returned instead'

    # Compute regret
    if v(solution) < 1.2:
        regret = 1
    else:
        regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret{regret}')


if __name__ == "__main__":
    main()

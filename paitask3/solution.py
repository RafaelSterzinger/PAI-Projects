import numpy as np
import gpytorch
import gpytorch.means as means
import gpytorch.kernels as kernels
import torch
from scipy.optimize import fmin_l_bfgs_b

domain = np.array([[0, 5]])

""" Solution """


class ExactGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, mean_module, covar_module):
        super().__init__(train_x, train_y, likelihood=gpytorch.likelihoods.GaussianLikelihood())
        self.mean_module = mean_module
        self.covar_module = covar_module

    def forward(self, x):
        """Forward computation of GP."""
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration. """
        self.min_v = 1.2
        self.xi = 0.02

        hypers_f = {
            'likelihood.noise': torch.tensor(0.15),
            'covar_module.base_kernel.lengthscale': torch.tensor(0.5),
            'covar_module.outputscale': torch.tensor(0.5),
        }
        self.gp_f = ExactGP(torch.Tensor(), torch.Tensor(), means.ZeroMean(),
                            kernels.ScaleKernel(kernels.MaternKernel(2.5)))
        self.gp_f.initialize(**hypers_f)
        self.gp_f.eval()
        self.gp_f.likelihood.eval()

        hypers_v = {
            'likelihood.noise': torch.tensor(0.0001),
            'covar_module.base_kernel.lengthscale': torch.tensor(0.5),
            'covar_module.outputscale': torch.tensor(np.sqrt(2)),
            'mean_module.constant': torch.tensor(1.5)
        }
        self.gp_v = ExactGP(torch.Tensor(), torch.Tensor(), means.ConstantMean(),
                            kernels.ScaleKernel(kernels.MaternKernel(2.5)))
        self.gp_v.initialize(**hypers_v)
        self.gp_v.eval()
        self.gp_v.likelihood.eval()

        self.init = False

        self.x = np.array([])
        self.f = np.array([])
        self.v = np.array([])

    def update_gp_f(self, new_inputs, new_targets):
        """Update GP with new points."""
        inputs = torch.cat((self.gp_f.train_inputs[0], new_inputs.unsqueeze(-1)), dim=0)
        targets = torch.cat((self.gp_f.train_targets, new_targets), dim=-1)
        self.gp_f.set_train_data(inputs, targets, strict=False)

    def update_gp_v(self, new_inputs, new_targets):
        """Update GP with new points."""
        inputs = torch.cat((self.gp_v.train_inputs[0], new_inputs.unsqueeze(-1)), dim=0)
        targets = torch.cat((self.gp_v.train_targets, new_targets), dim=-1)
        self.gp_v.set_train_data(inputs, targets, strict=False)

    def get_best_f(self):
        idx = self.gp_f.train_targets.argmax()
        return self.gp_f.train_inputs[0][idx], self.gp_f.train_targets[idx]

    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: np.ndarray
            1 x domain.shape[0] array containing the next point to evaluate
        """

        # In implementing this function, you may use optimize_acquisition_function() defined below.

        if not self.init:
            self.init = True
            return np.atleast_2d(domain[:, 0] + (domain[:, 1] - domain[:, 0]) * \
                 np.random.rand(domain.shape[0]))
        else:
            return self.optimize_acquisition_function()

    def optimize_acquisition_function(self):
        """
        Optimizes the acquisition function.

        Returns
        -------
        x_opt: np.ndarray
            1 x domain.shape[0] array containing the point that maximize the acquisition function.
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 30 times and pick best solution
        for _ in range(30):
            x0 = domain[:, 0] + (domain[:, 1] - domain[:, 0]) * \
                 np.random.rand(domain.shape[0])
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

        xmax, ymax = self.get_best_f()
        x = torch.from_numpy(x).float()
        out_f = self.gp_f(x)
        out_v = self.gp_v(x)
        dist = torch.distributions.Normal(torch.tensor([0.]), torch.tensor([1.]))
        Z_f = (out_f.mean - ymax - self.xi) / out_f.stddev
        Z_v = (out_v.mean - self.min_v - self.xi) / out_v.stddev
        impro_v = dist.cdf(Z_v)
        if((impro_v <= 0.5).item()):
            return impro_v.item()
        else:
            acquisition_function = ((out_f.mean - ymax - self.xi) * dist.cdf(Z_f) + out_f.stddev * torch.exp(
                dist.log_prob(Z_f))) * impro_v
            return acquisition_function.item()

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
        self.x = np.append(self.x, x.item())
        self.f = np.append(self.f, f)
        self.v = np.append(self.v, v)

        x = np.reshape(x, [1])
        f = np.reshape(f, [1])
        v = np.reshape(v, [1])
        self.update_gp_f(torch.Tensor(x), torch.Tensor(f))
        self.update_gp_v(torch.Tensor(x), torch.Tensor(v))

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: np.ndarray
            1 x domain.shape[0] array containing the optimal solution of the problem
        """

        mask = self.v >= self.min_v
        f_vals = self.f[mask]
        x_vals = self.x[mask]

        if len(f_vals) == 0:
            return self.optimize_acquisition_function()
        else:
            idx = f_vals.argmax()
            return x_vals[idx]


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
    for j in range(30):
        # Get next recommendation
        x = agent.next_recommendation()

        # Check for valid shape
        assert x.shape == (1, domain.shape[0]), \
            f"The function next recommendation must return a numpy array of " \
            f"shape (1, {domain.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x)
        cost_val = v(x)
        agent.add_data_point(x, obj_val, cost_val)

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
          f'{f(solution)}\nRegret {regret}')


if __name__ == "__main__":
    main()

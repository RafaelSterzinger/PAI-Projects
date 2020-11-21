import numpy as np
import gpytorch
import torch
import torch.nn as nn
from scipy.optimize import fmin_l_bfgs_b
from torch.distributions import Normal
import sys

domain = np.array([[0, 5]])

class ExactGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, mean_module, variance, likelihood=gpytorch.likelihoods.GaussianLikelihood()):
        
        likelihood.noise_covar.noise = variance #** 2
        super().__init__(train_x, train_y, likelihood=likelihood)
        self.mean_module = mean_module
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu = 5 / 2))
        self.covar_module.base_kernel.lengthscale = 0.5

    def forward(self, x):
        
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class BO_EI(nn.Module):
    """Abstract Bayesian Optimization class. ."""
    def __init__(self, gp_f, gp_v):
        
        super().__init__()
        
        self.xi = 0.01
        
        self.gp_f = gp_f
        self.gp_v = gp_v
        
        self.gp_f.eval()
        self.gp_f.likelihood.eval()
        
        self.gp_v.eval()
        self.gp_v.likelihood.eval()
        
    def update_gp (self, new_inputs, new_targets, is_gp_f):
        
        if is_gp_f:
            inputs = torch.cat([self.gp_f.train_inputs[0], new_inputs.unsqueeze(-1)], dim = 0)
            targets = torch.cat([self.gp_f.train_targets, new_targets], dim = -1)
            self.gp_f.set_train_data(inputs, targets, strict=False)
        else:
            inputs = torch.cat([self.gp_v.train_inputs[0], new_inputs.unsqueeze(-1)], dim = 0)
            targets = torch.cat([self.gp_v.train_targets, new_targets], dim = -1)
            self.gp_v.set_train_data(inputs, targets, strict=False)
    
    def get_best_value(self, is_gp_f):
        
        if is_gp_f:
            idx = self.gp_f.train_targets.argmax()
            xmax = self.gp_f.train_inputs[0][idx]
            ymax = self.gp_f.train_targets[idx]
        else:
            idx = self.gp_v.train_targets.argmax()
            xmax = self.gp_v.train_inputs[0][idx]
            ymax = self.gp_v.train_targets[idx]
        return xmax, ymax 
        
    def get_acquisition_function(self, x):
        
        xmax, ymax = self.get_best_value(True)
        xmax1, vmax = self.get_best_value(False)
        
        out_f = self.gp_f(x)
        out_v = self.gp_v(x)
        
        dist = Normal(torch.tensor([0.]), torch.tensor([1.]))
        Z1 = (out_f.mean - ymax - self.xi) / out_f.stddev 
        Z2 = (out_v.mean - vmax - self.xi) / out_v.stddev 
        
        acquisition_function = (out_f.mean - ymax - self.xi) * dist.cdf(Z1) + out_f.stddev * torch.exp(dist.log_prob(Z1)) * dist.cdf(Z2)
        return acquisition_function.detach().numpy()[0]
    

class BO_algo():
    
    def __init__(self):

        self.bo_ei = None
        
    def next_recommendation(self):

        if self.bo_ei is None:
            x0 = domain[:, 0] + (domain[:, 1] - domain[:, 0]) * np.random.rand(domain.shape[0]) #type [2.5]
            return np.array([x0])
        else:
            return self.optimize_acquisition_function()

    def optimize_acquisition_function(self):

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick best solution
        for _ in range(20):
            x0 = domain[:, 0] + (domain[:, 1] - domain[:, 0]) * np.random.rand(domain.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=domain, approx_grad=True)
            x_values.append(np.clip(result[0], *domain[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        return np.atleast_2d(x_values[ind])

    def acquisition_function(self, x):

        return self.bo_ei.get_acquisition_function(torch.from_numpy(x))


    def add_data_point(self, x, f, v):

        x = torch.from_numpy(x[0])
        f = torch.from_numpy(f)
        v = torch.tensor(np.log(v) - np.log(1.2))
        
        if self.bo_ei is None:
            
            mean = gpytorch.means.ConstantMean()
            mean.initialize(constant=1.5)
            mean.constant.requires_grad = False
            
            self.bo_ei = BO_EI(ExactGP(x, f, gpytorch.means.ZeroMean(), 0.5), ExactGP(x, v, mean, np.sqrt(2)))
            
        else:    
            self.bo_ei.update_gp(x, f, True)
            self.bo_ei.update_gp(x, v, False)
        
        
    def get_solution(self):

        mask = self.bo_ei.gp_v.train_targets.detach().numpy() >= 0
        f_vals = self.bo_ei.gp_f.train_targets.detach().numpy()[mask]
        x_vals = self.bo_ei.gp_f.train_inputs[0].detach().numpy()[mask]
        
        if len(f_vals) == 0:
            idx = self.bo_ei.gp_f.train_targets.detach().numpy().argmax()
            xmax = self.bo_ei.gp_f.train_inputs[0][idx]
        else:
            idx = f_vals.argmax()
            xmax = x_vals[idx]
        return xmax

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
    for j in range(20):
        # Get next recommendation
        x = agent.next_recommendation()
        # Check for valid shape
        assert x.shape == (1, domain.shape[0]),             f"The function next recommendation must return a numpy array of "             f"shape (1, {domain.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x)
        cost_val = v(x)
        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = np.atleast_2d(agent.get_solution())
    assert solution.shape == (1, domain.shape[0]),         f"The function get solution must return a numpy array of shape ("         f"1, {domain.shape[0]})"
    assert check_in_domain(solution),         f'The function get solution must return a point within the '         f'domain, {solution} returned instead'

    # Compute regret
    if v(solution) < 1.2:
        regret = 1
    else:
        regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret{regret}')


if __name__ == "__main__":
    main()

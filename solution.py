import numpy as np
import matplotlib.pyplot as plt
import gpytorch as gp
import torch
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

## Constant for Cost function
THRESHOLD = 0.5
W1 = 1
W2 = 20
W3 = 100
W4 = 0.04


def cost_function(true, predicted):
    """
        true: true values in 1D numpy array
        predicted: predicted values in 1D numpy array

        return: float
    """
    cost = (true - predicted) ** 2

    # true above threshold (case 1)
    mask = true > THRESHOLD
    mask_w1 = np.logical_and(predicted >= true, mask)
    mask_w2 = np.logical_and(np.logical_and(predicted < true, predicted >= THRESHOLD), mask)
    mask_w3 = np.logical_and(predicted < THRESHOLD, mask)

    cost[mask_w1] = cost[mask_w1] * W1
    cost[mask_w2] = cost[mask_w2] * W2
    cost[mask_w3] = cost[mask_w3] * W3

    # true value below threshold (case 2)
    mask = true <= THRESHOLD
    mask_w1 = np.logical_and(predicted > true, mask)
    mask_w2 = np.logical_and(predicted <= true, mask)

    cost[mask_w1] = cost[mask_w1] * W1
    cost[mask_w2] = cost[mask_w2] * W2

    reward = W4 * np.logical_and(predicted < THRESHOLD, true < THRESHOLD)
    if reward is None:
        reward = 0
    return np.mean(cost) - np.mean(reward)


"""
Fill in the methods of the Model. Please do not change the given methods for the checker script to work.
You can add new methods, and make changes. The checker script performs:


    M = Model()
    M.fit_model(train_x,train_y)
    prediction = M.predict(test_x)

It uses predictions to compare to the ground truth using the cost_function above.
"""


class ExactGPModel(gp.models.ExactGP):
    def __init__(self, train_x, train_y, kernel):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood=gp.likelihoods.GaussianLikelihood())
        self.mean_module = gp.means.ConstantMean()
        self.covar_module = get_kernel(kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gp.distributions.MultivariateNormal(mean_x, covar_x)


class Model():
    """
    We envisage that in order to solve this task you need to overcome three challenges - each requiring a specific strategy.

    1. Model selection - You will need to find the right kernel and its hyper-parameters that model the GP
    faithfully. With Bayesian models, a commonly used principle in choosing the right kernel or hyper-parameters is
    to use the "data likelihood", otherwise known as the marginal likelihood to find the best model. See more details
    at: Wiki
    2. Large scale learning - Natively, GP inference is computationally intensive for large datasets and
    common-place computers. The inference requires O(N3) basic operations in order find the posterior distributions.
    For large datasets this becomes infeasible. In order to solve this problem, practitioners use forms of low-rank
    approximations. The most popular are the Nyström method, using random features and/or other scalable
    clustering-based approaches. This excellent review on Wikipedia can serve as an introduction: Wiki.
    3. Asymmetric
    costs: We utilize a specifically designed cost function, where deviation from the true concentration levels is
    penalized, and you are rewarded for correctly predicting safe regions. Under this specific cost function,
    the mean prediction might not be optimal. Note that the mean prediction refers to the optimal decision with
    respect to a general squared loss and some posterior distribution over the true value to be predicted.
    """

    def predict(self, test_x):
        self.model.eval()
        test_x = torch.from_numpy(np.array(test_x)).float()
        y_preds = self.model.likelihood(self.model(test_x))
        sol_mean = y_preds.mean.data.numpy()
        sol_std = y_preds.stddev.data.numpy()

        return np.array(list(map(lambda x, y: min(x + y, 1), sol_mean, sol_std)))

    def fit_model(self, train_x, train_y):
        train_x = torch.from_numpy(np.array(train_x)).float()
        train_y = torch.from_numpy(np.array(train_y)).float()

        training_iter = 100

        best_kernel = {}
        for kernel in kernels:
            print("Start training with", kernel, "kernel")
            model = ExactGPModel(train_x, train_y, kernel)
            model.train()
            model.likelihood.train()

            # Use the adam optimizer
            optimizer = torch.optim.Adam([
                {'params': model.parameters()},  # Includes GaussianLikelihood parameters
            ], lr=0.1)
            # "Loss" for GPs - the marginal log likelihood
            mll = gp.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

            losses = []
            noise = []

            for i in range(training_iter):
                # Zero gradients from previous iteration
                optimizer.zero_grad()
                # Output from model
                output = model(train_x)
                # Calc loss and backprop gradients
                loss = -mll(output, train_y)
                loss.backward()

                losses.append(loss.item())
                noise.append(model.likelihood.noise.item())

                print('Iter %d/%d - Loss: %.3f   Noise: %.3f' % (
                    i + 1, training_iter, loss.item(),
                    model.likelihood.noise.item()
                ))
                optimizer.step()


            plt.plot(losses, label=f"{kernel}")
            plt.legend(loc="best")
            plt.xlabel("Num Iteration")
            plt.ylabel("MLL Loss")
            plt.show()
            best_kernel[kernel] = (loss.item(), model)

        best_model = min(best_kernel.values(), key=lambda x: x[0])[1]
        self.model = best_model


def visualize_data(train_x, train_y):
    train_x1 = list(map(lambda x: x[0], train_x))
    train_x2 = list(map(lambda x: x[1], train_x))
    label = list(map(lambda x: ('g' if x <= 0.5 else 'r'), train_y))
    fig = plt.figure()
    ax = Axes3D(fig)

    def init():
        ax.scatter(train_x1, train_x2, train_y, s=0.1, c=label, alpha=0.6)
        return fig,

    def animate(i):
        ax.view_init(elev=10., azim=i)
        return fig,

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=360, interval=20, blit=True)
    # Save
    anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])


# kernels = ["RBF", "linear", "quadratic", "Matern-1/2", "Matern-3/2", "Matern-5/2"]

# Best Model utilizes the "Matern-1/2" kernel with a loss of ~ -1.639
kernels = ["Matern-1/2"]


def get_kernel(kernel, composition="addition"):
    base_kernel = []
    if "RBF" in kernel:
        base_kernel.append(gp.kernels.RBFKernel())
    if "linear" in kernel:
        base_kernel.append(gp.kernels.LinearKernel())
    if "quadratic" in kernel:
        base_kernel.append(gp.kernels.PolynomialKernel(power=2))
    if "Matern-1/2" in kernel:
        base_kernel.append(gp.kernels.MaternKernel(nu=1 / 2))
    if "Matern-3/2" in kernel:
        base_kernel.append(gp.kernels.MaternKernel(nu=3 / 2))
    if "Matern-5/2" in kernel:
        base_kernel.append(gp.kernels.MaternKernel(nu=5 / 2))
    if "Cosine" in kernel:
        base_kernel.append(gp.kernels.CosineKernel())

    if composition == "addition":
        base_kernel = gp.kernels.AdditiveKernel(*base_kernel)
    elif composition == "product":
        base_kernel = gp.kernels.ProductKernel(*base_kernel)
    else:
        raise NotImplementedError
    kernel = gp.kernels.ScaleKernel(base_kernel)
    return kernel


def main():
    # load the train dateset
    train_x_name = "train_x.csv"
    train_y_name = "train_y.csv"

    train_x = np.loadtxt(train_x_name, delimiter=',')
    train_y = np.loadtxt(train_y_name, delimiter=',')

    # load the test dateset
    test_x_name = "test_x.csv"
    test_x = np.loadtxt(test_x_name, delimiter=',')

    M = Model()
    M.fit_model(train_x, train_y)
    prediction = M.predict(test_x)

    print(prediction)


if __name__ == "__main__":
    main()

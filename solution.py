import numpy as np
import matplotlib.pyplot as plt

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

    def __init__(self):
        """
            TODO: enter your code here
        """
        pass

    def predict(self, test_x):
        """
            TODO: enter your code here
        """
        ## dummy code below
        y = np.ones(test_x.shape[0]) * THRESHOLD - 0.00001
        return y

    def fit_model(self, train_x, train_y):
        """
             TODO: enter your code here
        """
        pass


def vis(train_x, train_y):
    train_x1 = list(map(lambda x: x[0], train_x))
    train_x2 = list(map(lambda x: x[1], train_x))
    label = list(map(lambda x: ('g' if x <= 0.5 else 'r'), train_y))
    plt.scatter(train_x1, train_x2, s=0.1, c=label)
    plt.show()

def main():
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

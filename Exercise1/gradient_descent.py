# used for manipulating directory paths
import os

# Scientific and vector computation for python
import numpy as np

# Plotting library
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D  # needed to plot 3-D surfaces

# library written for this exercise providing additional functions for assignment submission, and others
import utils

# Read comma separated data
data = np.loadtxt(os.path.join('Data', 'ex1data1.txt'), delimiter=',')
X, y = data[:, 0], data[:, 1]

m = y.size  # number of training examples

# Add a column of ones to X. The numpy function stack joins arrays along a given axis.
# The first axis (axis=0) refers to rows (training examples)
# and second axis (axis=1) refers to columns (features).
X = np.stack([np.ones(m), X], axis=1)


def plotData(x, y):
    """
    Plots the data points x and y into a new figure. Plots the data
    points and gives the figure axes labels of population and profit.

    Parameters
    ----------
    x : array_like
        Data point values for x-axis.

    y : array_like
        Data point values for y-axis. Note x and y should have the same size.

    Instructions
    ------------
    Plot the training data into a figure using the "figure" and "plot"
    functions. Set the axes labels using the "xlabel" and "ylabel" functions.
    Assume the population and revenue data have been passed in as the x
    and y arguments of this function.

    Hint
    ----
    You can use the 'ro' option with plot to have the markers
    appear as red circles. Furthermore, you can make the markers larger by
    using plot(..., 'ro', ms=10), where `ms` refers to marker size. You
    can also set the marker edge color using the `mec` property.
    """
    fig = pyplot.figure()  # open a new figure

    # ====================== YOUR CODE HERE =======================
    pyplot.plot(x, y, 'ro', ms=10, mec='k')
    pyplot.ylabel('Profit in $10,000')
    pyplot.xlabel('Population of City in 10,000s')
    # =============================================================


def ht(x, t):
    hypothesis = 0
    for i in range(len(x)):
        hypothesis += t[i] * x[i]
    return hypothesis


def computeCost(X, y, theta):
    """
    Compute cost for linear regression. Computes the cost of using theta as the
    parameter for linear regression to fit the data points in X and y.

    Parameters
    ----------
    X : array_like
        The input dataset of shape (m x n+1), where m is the number of examples,
        and n is the number of features. We assume a vector of one's already
        appended to the features so we have n+1 columns.

    y : array_like
        The values of the function at each data point. This is a vector of
        shape (m, ).

    theta : array_like
        The parameters for the regression function. This is a vector of
        shape (n+1, ).

    Returns
    -------
    J : float
        The value of the regression cost function.

    Instructions
    ------------
    Compute the cost of a particular choice of theta.
    You should set J to the cost.
    """

    # initialize some useful values
    m = y.size  # number of training examples

    # You need to return the following variables correctly
    J = 0

    # ====================== YOUR CODE HERE =====================

    J = 0
    for i in range(m):
        J += (ht(X[i], theta) - y[i]) ** 2
    J = J * 1 / (2 * m)

    # ===========================================================
    return J


# J = computeCost(X, y, theta=np.array([0.0, 0.0]))
# print('With theta = [0, 0] \nCost computed = %.2f' % J)
# print('Expected cost value (approximately) 32.07\n')
#
# # further testing of the cost function
# J = computeCost(X, y, theta=np.array([-1, 2]))
# print('With theta = [-1, 2]\nCost computed = %.2f' % J)
# print('Expected cost value (approximately) 54.24')


def gradientDescent(X, y, thetas, alpha, num_iters):
    """
    Performs gradient descent to learn `theta`. Updates theta by taking `num_iters`
    gradient steps with learning rate `alpha`.

    Parameters
    ----------
    X : array_like
        The input dataset of shape (m x n+1).

    y : arra_like
        Value at given features. A vector of shape (m, ).

    thetas : array_like
        Initial values for the linear regression parameters.
        A vector of shape (n+1, ).

    alpha : float
        The learning rate.

    num_iters : int
        The number of iterations for gradient descent.

    Returns
    -------
    theta : array_like
        The learned linear regression parameters. A vector of shape (n+1, ).

    J_history : list
        A python list for the values of the cost function after each iteration.

    Instructions
    ------------
    Peform a single gradient step on the parameter vector theta.

    While debugging, it can be useful to print out the values of
    the cost function (computeCost) and gradient here.
    """
    # Initialize some useful values
    m = y.shape[0]  # number of training examples

    # make a copy of theta, to avoid changing the original array, since numpy arrays
    # are passed by reference to functions
    thetas = thetas.copy()

    J_history = []  # Use a python list to save cost in every iteration

    temp_theta = []

    for iteration in range(num_iters):
        # ==================== YOUR CODE HERE =================================

        # run cost function on all samples for each theta
        for idx, theta in enumerate(thetas):

            dJdt = 0
            for i in range(m):
                dJdt += (ht(X[i], thetas) - y[i]) * X[i][idx]

            newTheta = theta - alpha * 1/m * dJdt #(dJdt * 1 / (2 * m))
            temp_theta.insert(idx, newTheta)

        # simultaneously update thetas for all J
        print("Iteration: " + str(iteration))
        for idx,t in enumerate(thetas):
            print("Theta" + str(idx), temp_theta[idx])
            thetas[idx] = temp_theta[idx]
        # =====================================================================

        # save the cost J in every iteration
        J_history.append(computeCost(X, y, thetas))

    return thetas, J_history

# initialize fitting parameters
theta = np.zeros(2)

# some gradient descent settings
iterations = 1500
alpha = 0.01

theta, J_history = gradientDescent(X ,y, theta, alpha, iterations)
print('Theta found by gradient descent: {:.4f}, {:.4f}'.format(*theta))
print('Expected theta values (approximately): [-3.6303, 1.1664]')


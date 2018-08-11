# Title: ex1.py
# Author: Jun Hao Hu, University of California San Diego
# Date: August 08, 2018
# Purpose: Python procedure that trains a feed-forward neural network for the purpose of solving a
# first-order ODE.

# Import statements

import numpy as np

from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import autograd.numpy as np
from autograd import grad, jacobian
import autograd.numpy.random as npr

import scipy.io as sio

# Preliminary definitions

nx = 41

dx = 1./nx

x_space = np.linspace(1,3,nx)

'''
Define a function that returns the value of the analytic solution at a specified point.
'''

analytic_solution = lambda input_point : input_point*np.tan(np.log(input_point))

'''
Define a function that returns the value of the leaky rectified linear unit function, with alpha = 0.01, at a specified point.
'''

relu = lambda input_point : np.maximum(0.01*input_point,input_point)

'''
Define a function that returns the value of the sigmoid function, at a specified point.
'''

sigmoid = lambda input_point : 1./(1+np.exp(-input_point))

'''
Define a function that returns the value of the derivative of the leaky rectified linear unit function, at a specified point.
'''

relu_prime = lambda input_point : 1 if input_point > 0 else 0.01

'''
Define the function xi, which is part of the trial function, evaluated at a specific point.
'''

xi = lambda input_point : (input_point-1)

'''
Define the function gamma, which is part of the trial function, evaluated at a specific point.
'''

gamma = lambda input_point : (input_point-1)**2

'''
Using the above two functions, define the trial solution evaluated at a specific point.
'''

psi_solution = lambda input_point, network_params : xi(input_point)+gamma(input_point)*network_params

'''
Define the function f, which is the right hand side of the ODE.
'''

f = lambda input_point, psi : 1+(psi/input_point)+(psi/input_point)**2

'''
Define neural network operation.
'''

def neural_network(W,x):
    a1 = sigmoid(np.dot(x,W[0]))
    return np.dot(a1,W[1])

'''
Define another neural network operation, this time a version that does not require us to provide the weights.
'''

def neural_network_x(x):
    a1 = sigmoid(np.dot(x,W[0]))
    return np.dot(a1,W[1])

'''
Define the loss function.
'''

def cost_function(W,x):
    cost_sum = 0.
    psi_grad = grad(psi_solution)

    for ix in x:
        net_out = neural_network(W,ix)[0][0]

        net_out_grad = grad(neural_network_x)(ix)

        psi_t = psi_solution(ix,net_out)

        grad_of_psi = psi_grad(ix,net_out)

        func = f(ix,psi_t)

        err_squared = np.abs(grad_of_psi - func)
        cost_sum += err_squared
    return cost_sum

'''
Main meat of the program.
'''

W = [npr.randn(1,nx), npr.randn(nx,1)]
lmb = 0.0001

for i in range(5000):
    loss_grad = grad(cost_function)(W,x_space)

    W[0] -= lmb*loss_grad[0]
    W[1] -= lmb*loss_grad[1]

print(cost_function(W,x_space))
res = [psi_solution(ix,neural_network(W,ix)[0][0]) for ix in x_space]
#rk4_res = sio.loadmat('./ex1.mat')['w']
#rk4_res = rk4_res.flatten()

plt.figure()
plt.plot(x_space,analytic_solution(x_space))
plt.plot(x_space,res)
#plt.plot(x_space,rk4_res)
plt.show()

plt.figure()
plt.plot(x_space,np.abs(analytic_solution(x_space)-res))
#plt.plot(x_space,np.abs(analytic_solution(x_space)-rk4_res))
plt.show()

# the solution is a solution to
# - grad(a(x) grad u) + |grad u|^2 = f, in a domain where a(x) = 1+1/2*|x|^2

import torch
import numpy
from numpy import sin, cos, zeros, pi, sqrt, absolute, exp, array
from torch import optim, autograd, Tensor

torch.set_default_tensor_type('torch.cuda.FloatTensor')

h = 0.001 # step length ot compute derivative, truncation error is O(d^-5)


# the dimension in space， should be at least 2
def spatial_dimension():
    return 2


# input ny array, output pointwise true solution of the pde, np array
def true_solution(x):
    r = sqrt(numpy.sum(x[:,1:]**2,1))
    temp = absolute(1-r)
    
    u = (exp(x[:,0]**2)-1)*sin(pi/2*temp**(2.5))
    return u


# the point-wise nabla operator, derivative of output w.r.t. inputs, both input tensor, output tensor
def gradients(output, inputs):
    return autograd.grad(outputs=output, inputs=inputs,
                                grad_outputs=torch.ones_like(output),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]


# the point-wise Du: Input x is a numpy array of points ; Output is a numpy vector which means the result of Du(x))
def Du(network_solution, x, d):
    dudx = autograd.grad(outputs= network_solution, inputs=x,
                              grad_outputs=torch.ones_like(network_solution),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    Du = gradients(dudx[:, 0: 1], x)[:, 0: 1]
    for i in range(1, d+1):
        Du = Du - gradients(dudx[:, i: i+1], x)[:, i: i+1]
    return Du


# the point-wise Du: Input x is a numpy array of points ; Output is a numpy vector which means the result of Du(x))
# Du above that uses Autograd is not outputting correct second partial derivatives, so we use numerical differentiation here.
def Du_numerical(model,tensor_x):
    modelx = model(tensor_x)
    s = torch.zeros(tensor_x.shape[0],)
    for i in range(tensor_x.shape[1]):
        ei = torch.zeros(tensor_x.shape)
        ei[:,i] = 1
        if i == 0:
            s = s + (model(tensor_x+h*ei)-2*modelx+model(tensor_x-h*ei))/h/h
        else:
            s = s - (model(tensor_x+h*ei)-2*modelx+model(tensor_x-h*ei))/h/h
    return s


# define the right hand function for input numpy array, output numpy array
def f(x):
    d = x.shape[1]-1
    r = sqrt(numpy.sum(x[:,1:]**2,1))
    temp = absolute(1-r)
    inner_part = pi/2*temp**(2.5)
    u_tilde = sin(inner_part)
    Laplace_u_tilde = -5*pi*(d-1)/4/r*cos(inner_part)*temp**1.5-25*pi*pi/16*sin(inner_part)*(1-r)**3+15*pi/8*cos(inner_part)*temp**0.5
    
    f = 2*(1+2*x[:,0]**2)*exp(x[:,0]**2)*u_tilde-(exp(x[:,0]**2)-1)*Laplace_u_tilde
    return f


# the point-wise Bu output tensor, input torch tensor
def Bu(model,tensor_x):
    return model(tensor_x)


# define the boundary value g output tensor, input torch tensor
def g(x):
    return zeros((x.shape[0],))


# the point-wise h0 input numpy array/torch tensor output numpy array
def h0(x):
    return zeros((x.shape[0],))
    
    
# the point-wise h1 input numpy array/torch tensor output numpy array
def h1(x):
    return zeros((x.shape[0],))


# the point-wise dudt, the initial condition, input torch tensor, output torch tensor
def partial_u_partial_t(network_solution, x):
    first_derivative = autograd.grad(outputs=network_solution, inputs=x,
                               grad_outputs=torch.ones_like(network_solution),
                               create_graph=True, retain_graph=True, only_inputs=True)[0]
    dudt = first_derivative[:, 0: 1]
    return dudt


# output the domain parameters
def domain_parameter():
    R = 1
    return R


# output the time interval
def time_interval():
    return array([0,1])


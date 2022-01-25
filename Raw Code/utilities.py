import numpy
import matplotlib
import torch


# generate uniform distributed points in the sphere [T0,T1] X {x:|x|<R}, output np array
# input d is the dimension， R is radius of sphere domain, N is number of points generated
def generate_uniform_points_in_sphere_time_dependent(d,R,time_interval,N):
    points = numpy.zeros((N,d+1))
    points[:,0] = numpy.random.uniform(time_interval[0],time_interval[1],(N,))
    points[:,1:] = numpy.random.normal(size=(N,d))
    scales = (numpy.random.uniform(0,1,(N,)))**(1/d)
    for i in range(N):
        points[i,1:] = points[i,1:]/numpy.sqrt(numpy.sum(points[i,1:]**2))*scales[i]*R
    return points


# generate uniform distributed points on the boundary of time-dependent domain [T0,T1]X{|x|<R}, output np array
# d is dimension, N_bdy is number of points on bdy, N_ini is number of point on initial slice
def generate_uniform_points_on_sphere_time_dependent(d,R,time_interval,N_boundary,N_initial_time):
    points_bd = numpy.zeros((N_boundary,d+1))
    points_int = numpy.zeros((N_initial_time,d+1))
    points_int[:,0] = time_interval[0]*numpy.ones(N_initial_time,)
    points_int[:,1:] = numpy.random.normal(size=(N_initial_time,d))
    for i in range(N_boundary):
        points_bd[i,0] = numpy.random.uniform(time_interval[0],time_interval[1])
        points_bd[i,1:] = numpy.random.normal(size=(1,d))
        points_bd[i,1:] = points_bd[i,1:]/numpy.sqrt(numpy.sum(points_bd[i,1:]**2))*R
    points_int[:,0] = time_interval[0]*numpy.ones(N_initial_time,)
    points_int[:,1:] = numpy.random.normal(size=(N_initial_time,d))
    scales = (numpy.random.uniform(0,R,(N_initial_time,)))**(1/d)
    for i in range(N_initial_time):
        points_int[i,1:] = points_int[i,1:]/numpy.sqrt(numpy.sum(points_int[i,1:]**2))*scales[i]
    return points_bd, points_int


# input 2 torch tensors, output a torch tensor, l2 error of the two input tensor
def get_l2_error(network_solution, true_solution):
    return torch.norm(network_solution - true_solution) / torch.norm(true_solution)
    

# output grid points (t = 0.02i, x = 0.05j, y = 0.05k, for i, j, k integers s.t. t in [0,1], |(x,y)|<R) in other words, grid points in the domain, np array
# this is only for 2d case
def grid_points_for_2d_only():
    t_coor = numpy.arange(0.02, 1, 0.02)
    x_coor = numpy.arange(-0.95, 1, 0.05)
    y_coor = numpy.arange(-0.95, 1, 0.05)
    grid_pts = numpy.array([[t, x, y] for t in t_coor for x in x_coor for y in y_coor])
    grid_pts = grid_pts[grid_pts[:,1]**2 + grid_pts[:, 2]**2 < 1]
    return grid_pts



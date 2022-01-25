import torch
from torch import Tensor, optim
import numpy
import scipy.io
from numpy import zeros, sum, sqrt, linspace, absolute
from utilities import generate_uniform_points_in_sphere_time_dependent,generate_uniform_points_on_sphere_time_dependent, \
                            get_l2_error, grid_points_for_2d_only
import network_setting
import time
import timeit
from problem_setting import spatial_dimension, true_solution, Du, Du_numerical, Bu, f, g, h0,\
    domain_parameter, time_interval, partial_u_partial_t

######################### Preliminary Settings ###########################
torch.manual_seed(2)   # fix random seed, so that results are replicable
numpy.random.seed(2)
use_cuda = torch.cuda.is_available()   # check if cuda on machine, if yes, run on GPU, set to double precision, ieee standard
device = torch.device("cuda" if use_cuda else "cpu")
torch.set_default_tensor_type(torch.DoubleTensor)

################### PDE Specific Settings ########################
d = spatial_dimension()  # dimension of problem
R = domain_parameter()   # R = 1, sphere domain
time_interval = time_interval()  # time interval is [0, 1]
lambda_term = 300      # lambda term for balancing bdy and initial conditions

#################### Training Specific Settings #####################
n_epoch = 2000  # number of outer iterations
N_interior = 1024 # number of training sampling points inside the domain in each epoch (batch size)
N_boundary = 1024
N_initial = 512
N_ini_bdy = N_boundary + N_initial
lr = 10e-3 # initial learning rate

################### Interface Settings ######################
n_epoch_show_info = max([round(n_epoch/50),1])      # training information will be shown per n_epoch_show_info epochs

####################### Network Setting  ######################
m = 50       # number of nodes in each hidden layer
solution_net = network_setting.network_time_depedent(d, m)
optimizer = optim.Adam(solution_net.parameters(), lr=lr)
StepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.99)

############# Storing Data ####################
lossseq = zeros((n_epoch,))
l2errorseq = zeros((n_epoch,))
time_seq = zeros((n_epoch,))


############ Error Measurement ############
x_test = grid_points_for_2d_only()
true_sol_x_test = true_solution(x_test)


############ Training ###############    
k = 0       
time = 0
while k < n_epoch:
    tic = timeit.default_timer()
    ################# Setting up the Loss Functional ###################
    x_interior = generate_uniform_points_in_sphere_time_dependent(d,R,time_interval, N_interior)
    x_boundary, x_initial = generate_uniform_points_on_sphere_time_dependent(d,R,time_interval, N_boundary, N_initial)
    tensor_x_interior = Tensor(x_interior)
    tensor_x_interior.requires_grad_()
    tensor_f = Tensor(f(x_interior))
    tensor_x_boundary = Tensor(x_boundary)
    tensor_x_boundary.requires_grad=False
    tensor_g = Tensor(g(x_boundary))
    tensor_g.requires_grad=False
    tensor_x_initial = Tensor(x_initial)
    tensor_x_initial.requires_grad= True
    tensor_h = Tensor(h0(x_initial))
    tensor_h.requires_grad = False
    loss1 = torch.mean((Du_numerical(solution_net, tensor_x_interior) - tensor_f) ** 2)
    # loss1 = torch.mean(torch.square(Du(solution_net(tensor_x_interior), tensor_x_interior, d) - tensor_f))
    loss2 = torch.sum((Bu(solution_net,tensor_x_boundary)-tensor_g)**2)
    loss2 = loss2 + torch.sum((solution_net(tensor_x_initial)-tensor_h)**2)
    loss2 = loss2 + torch.sum((partial_u_partial_t(solution_net(tensor_x_initial),tensor_x_initial)-tensor_h)**2)
    loss2 = lambda_term/N_ini_bdy*loss2
    loss = loss1 + loss2

    ################ Updating network parameters/ Adam Step ##############
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    StepLR.step()
    toc = timeit.default_timer()
    time_one_epoch = toc - tic
    time = time + time_one_epoch
    time_seq[k] = time
    # Save loss and L2 error
    lossseq[k] = loss.item()
    tensor_x_test = Tensor(x_test)
    tensor_u_x_net = solution_net(tensor_x_test)
    tensor_true_sol_x_test = Tensor(true_sol_x_test)
    l2error = get_l2_error(tensor_u_x_net, tensor_true_sol_x_test)
    l2errorseq[k] = l2error
    ## Print information
    if k%n_epoch_show_info==0:
        print("epoch = %d, loss = %2.6f" %(k,loss.item()), end='')
        print(", loss1 = %2.6f" %loss1, end='')
        print(", l2 error = %2.6e" % l2error, end='')
        print(", time= %2.1f" %time, end='')
        print("\n")
    # increment
    k = k + 1

# Save solution_net
scipy.io.savemat('testing_l2errorseq.mat', mdict={'l2errorseq': l2errorseq})
scipy.io.savemat('time_seq.mat', mdict={'time_seq': time_seq})
torch.save(solution_net.state_dict(), 'networkpara'+'.pt')

# empty the GPU memory
torch.cuda.empty_cache()

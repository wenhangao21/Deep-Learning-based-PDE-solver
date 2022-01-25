import torch
import numpy
import network_setting
from numpy import sin, cos, zeros, pi, sqrt, absolute, exp, array


################## Set Points for which you want the solution ##########
pts = numpy.array([[0.8, 0.4, 0.3], [0.9, 0.3, 0.2]])   # pts should be a numpy array of shame n by 3.

######################### Preliminary Settings ###########################
# fix random seed, so that results are replicable
torch.manual_seed(2)
numpy.random.seed(2)
# check if cuda on machine, if yes, run on GPU, set to double precision, ieee standard
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.set_default_tensor_type(torch.DoubleTensor)
######### Creating Grid Points for Visualization #############
d = 2
m = 50
solution_net = network_setting.network_time_depedent(d, m)
solution_net.load_state_dict(torch.load('networkpara.pt'))
net_solution = solution_net.predict(pts)   # network solution, numpy array contains corresponding network solutions
print("the network prediction is: ",net_solution, "\n")

def true_solution(x):
    r = sqrt(numpy.sum(x[:, 1:] ** 2, 1))
    temp = absolute(1 - r)

    u = (exp(x[:, 0] ** 2) - 1) * sin(pi / 2 * temp ** (2.5))
    return u

print("the true solution is is: ",true_solution(pts), "\n")

# empty the GPU memory
torch.cuda.empty_cache()
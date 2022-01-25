import torch
from torch import Tensor, optim
import numpy
import network_setting
from problem_setting import true_solution
import scipy.io


######################### Preliminary Settings ###########################
# fix random seed, so that results are replicable
torch.manual_seed(2)
numpy.random.seed(2)
# check if cuda on machine, if yes, run on GPU, set to double precision, ieee standard
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.set_default_tensor_type(torch.DoubleTensor)
######### Creating Grid Points for Visualization #############
def grid_points():
    t_coor = numpy.arange(0.02, 1, 0.02)
    x_coor = numpy.arange(-0.96, 1, 0.04)
    y_coor = numpy.arange(-0.96, 1, 0.04)
    grid_pts = numpy.array([[t, x, y] for t in t_coor for x in x_coor for y in y_coor])
    grid_pts = grid_pts[grid_pts[:,1]**2 + grid_pts[:, 2]**2 < 1]
    return grid_pts


d = 2
m = 50
solution_net = network_setting.network_time_depedent(d, m)
solution_net.load_state_dict(torch.load('networkpara.pt'))
grid_pts = grid_points()
net_solution = solution_net(Tensor(grid_pts))
net_sol = net_solution.cpu().detach().numpy()
true_sol = true_solution(grid_pts)
scipy.io.savemat('grid_points.mat', mdict={'grid_points': grid_pts})
scipy.io.savemat('net_solution.mat', mdict={'net_solution': net_sol})
scipy.io.savemat('true_solution.mat', mdict={'true_solution': true_sol})

# empty the GPU memory
torch.cuda.empty_cache()
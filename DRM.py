import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim, autograd

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pylab as pylt


from dolfin import *
import mshr

import numpy as np
import pandas as pd
import math

import os
import time


parameters['allow_extrapolation'] = True
parameters["form_compiler"]["optimize"] = True  # optimize compiler options
parameters["form_compiler"]["cpp_optimize"] = True # optimize code when compiled in c++
set_log_active(False)  # handling of log messages, warnings and errors.


# Try to solve the poisson equation:
'''  Solve the following PDE
-\Delta u(x) = 0, x\in \Omega,
u(r,theta) = r^(2/3)sin(2/3*pi), x\in \partial \Omega
\Omega = (-1,1) * (-1,1) \[[0,1) * (-1,0]]
'''

# Expression for exact solution


class Expression_u(UserExpression):

    def __init__(self, omega, **kwargs):
        super().__init__(**kwargs)  # This part is new!
        self.omega = omega

    def eval(self, value, x):

        r = sqrt(x[0]*x[0] + x[1]*x[1])
        theta = math.atan2(abs(x[1]), abs(x[0]))

        if x[0] < 0 and x[1] > 0:
            theta = pi - theta

        elif x[0] <= 0 and x[1] <= 0:
            theta = pi + theta

        if r == 0.0:
            value[0] = 0.0
        else:
            value[0] = pow(r, pi/self.omega)*sin(theta*pi/self.omega)

    def eval_at_point(self, x):

        r = np.sqrt(x[0]*x[0] + x[1]*x[1])
        theta = math.atan2(abs(x[1]), abs(x[0]))

        if x[0] < 0 and x[1] > 0:
            theta = pi - theta

        elif x[0] <= 0 and x[1] <= 0:
            theta = pi + theta

        if r == 0.0:
            value = 0.0
        else:
            value = pow(r, pi/self.omega)*sin(theta*pi/self.omega)

        return value

    def value_shape(self):
        return ()


class PowerReLU(nn.Module):
    """
    Implements simga(x)^(power)
    Applies a power of the rectified linear unit element-wise.

    NOTE: inplace may not be working.
    Can set inplace for inplace operation if desired.
    BUT I don't think it is working now.

    INPUT:
        x -- size (N,*) tensor where * is any number of additional
             dimensions
    OUTPUT:
        y -- size (N,*)
    """

    def __init__(self, inplace=False, power=3):
        super(PowerReLU, self).__init__()
        self.inplace = inplace
        self.power = power

    def forward(self, input):
        y = F.relu(input, inplace=self.inplace)
        return torch.pow(y, self.power)



class Block(nn.Module):
    """
    IMplementation of the block used in the Deep Ritz
    Paper

    Parameters:
    in_N  -- dimension of the input
    width -- number of nodes in the interior middle layer
    out_N -- dimension of the output
    phi   -- activation function used
    """

    def __init__(self, in_N, width, out_N, phi=PowerReLU()):
        super(Block, self).__init__()
        # create the necessary linear layers
        self.L1 = nn.Linear(in_N, width)
        self.L2 = nn.Linear(width, out_N)
        # choose appropriate activation function
        self.phi = nn.Tanh()

    def forward(self, x):
        return self.phi(self.L2(self.phi(self.L1(x)))) + x


class drrnn(nn.Module):
    """
    drrnn -- Deep Ritz Residual Neural Network

    Implements a network with the architecture used in the
    deep ritz method paper

    Parameters:
        in_N  -- input dimension
        out_N -- output dimension
        m     -- width of layers that form blocks
        depth -- number of blocks to be stacked
        phi   -- the activation function
    """

    def __init__(self, in_N, m, out_N, depth=4, phi=PowerReLU()):
        super(drrnn, self).__init__()
        # set parameters
        self.in_N = in_N
        self.m = m
        self.out_N = out_N
        self.depth = depth
        self.phi = nn.Tanh()
        # list for holding all the blocks
        self.stack = nn.ModuleList()

        # add first layer to list
        self.stack.append(nn.Linear(in_N, m))

        # add middle blocks to list
        for i in range(depth):
            self.stack.append(Block(m, m, m))

        # add output linear layer
        self.stack.append(nn.Linear(m, out_N))

    def forward(self, x):
        # first layer
        for i in range(len(self.stack)):
            x = self.stack[i](x)
        return x


def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)




def get_interior_points(N=150, d=2):
    """
    randomly sample N points from interior of [-1,1]^d
    """
    # return points for each block
    n = N//3

    x = (torch.rand(2*n)*2 - 1).unsqueeze(-1)
    y = torch.rand(2*n).unsqueeze(-1)
    X1 = torch.cat((x, y), dim=1)

    X2 = -torch.rand(n, 2)
    X_final = torch.cat((X1, X2), dim=0)

    return X_final



def get_boundary_points(N=33):
    index = torch.rand(N, 1)
    index1 = torch.rand(N, 1) * 2 - 1
    # x in (0,1) y = 0
    xb1 = torch.cat((index, torch.zeros_like(index)), dim=1)
    # x = 1 y in (0,1)
    xb2 = torch.cat((torch.ones_like(index1), index), dim=1)
    # x in (-1,1) y = 1
    xb3 = torch.cat((index1, torch.ones_like(index)), dim=1)
    # x = -1 y in (-1,1)
    xb4 = torch.cat((torch.full_like(index1, -1), index1), dim=1)
    # x in (-1,0) y = -1
    xb5 = torch.cat((-index, torch.full_like(index1, -1)), dim=1)
    # x = 0 y in (-1,0)
    xb6 = torch.cat((torch.full_like(index1, 0), -index), dim=1)
    xb = torch.cat((xb1, xb2, xb3, xb4, xb5, xb6), dim=0)

    return xb




def get_interior_boundary_mesh(coords):

    xr = []
    xb = []

    for z in coords:
        if z[0] >= 0.99 or z[1] >= 0.99 or z[0] <= -0.99 or z[1] <= -0.99:
            xb.append(z)

        elif(near(z[1], 0) and z[0] >= 0):
            #z[1] = 0.0
            xb.append(z)

        elif(near(z[0], 0) and z[1] <= 0):
            #z[0] = 0.0
            xb.append(z)
        else:
            xr.append(z)

    xr = np.array(xr)
    xb = np.array(xb)

    xr = torch.tensor(xr).float()
    xb = torch.tensor(xb).float()

    return xr,xb




def inside_domain(Z):
    
    '''
        used for plotting solution with matplotlib
    '''
    
    Z_f = Z.clone()

    for i, z in enumerate(Z):
        if (z[0] > 1e-16) & (z[1] < 1e-16):
            Z_f[i] = np.nan

    return Z_f


def main(beta_,dof_):
    
    
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device used is: ',device)
    
    plt_exact = False
    plt_pred = True
    OT_mesh = True
    eval_OT = False
    save_mesh = False
    save_sol = False
    training = False
    omega = 3/2*pi
    domain_vertices = [Point(0.0, 0.0),
                       Point(1.0, 0.0),
                       Point(1.0, 1.0),
                       Point(-1.0, 1.0),
                       Point(-1.0, -1.0), Point(0.0, -1.0)]
    
    dof_OT = dof_
    beta= beta_
    
    max_epochs = 50000
    lr=1e-4
    in_N = 2
    m = 20
    out_N = 1
    depth = 4

    
    string_mesh = f'mesh_OT/mesh_OT_0.44_{dof_OT}.xml.gz'
    mesh = Mesh(string_mesh)
       
    
    geometry = mshr.Polygon(domain_vertices)
    df = pd.read_csv('coords_to_res.csv')
    if dof_OT in df['coords'].values:
        res = df[df['coords'] == dof_OT]['res']
    else:
        while dof_OT not in df['coords'].values:
            dof_OT += 1
            res = df[df['coords'] == dof_OT]['res']
    
    mesh_del = mshr.generate_mesh(geometry, res)
                
    # Define function space for Delaunay mesh 
    if eval_OT:
        mesh_eval = Mesh(mesh)
    else:
        mesh_eval = Mesh(mesh_del)
    
    V = FunctionSpace(mesh_eval, "CG", 1)  
    u = Function(V)
    u_best = Function(V)
    coords = V.tabulate_dof_coordinates()

    u_exp = Expression_u(omega, degree=5)
    
    xr_mesh,xb_mesh = get_interior_boundary_mesh(coords)
    Nr = len(xr_mesh)
    Nb = len(xb_mesh)
    
        
    if plt_exact:    
        u = interpolate(u_exp, V)
        pylt.figure(figsize=(10, 10))
        p = plot(u)
        # set colormap
        p.set_cmap("coolwarm")
        pylt.colorbar(p)
        pylt.xlabel('x')
        pylt.ylabel('y')
        pylt.show()
            
    
    full_path = os.path.realpath(__file__)
    path, filename = os.path.split(full_path)
    var_name = f'/Nets_DRM/dof_{dof_OT}_m_{m}_depth_{depth}_beta_{beta}_lr_{lr}_epochs_{max_epochs}/Var/vars_coll_{OT_mesh}'
    net_name = f'/Nets_DRM/dof_{dof_OT}_m_{m}_depth_{depth}_beta_{beta}_lr_{lr}_epochs_{max_epochs}/Net_coll_{OT_mesh}'
    var_model = path + var_name
    net_model = path + net_name
    os.makedirs(var_model, exist_ok=True)
    os.makedirs(net_model, exist_ok=True)
    
    
    if save_mesh:
        File(net_model + '/Mesh/mesh_OT.pvd') << mesh
        File(net_model + '/Mesh/mesh_Delaunay.pvd') << mesh_eval
   
    model = drrnn(in_N, m, out_N, depth=4).to(device)
    model.apply(weights_init)
    optimizer = optim.Adam(model.parameters(), lr)

    best_epoch = 0
    best_loss = 1e5
    
    if OT_mesh:   
        V = FunctionSpace(mesh, "CG", 1)  
        coords_OT = V.tabulate_dof_coordinates()
        xr_OT, xb_OT = get_interior_boundary_mesh(coords_OT)
        ncells = mesh.num_cells()
        area_dict = torch.zeros(ncells).to(device)
        mid_vertex_dict = np.zeros((ncells,3,2))

        for cell in cells(mesh):
                idx = cell.index()
                tri_area = cell.volume()
                vertex_coords = np.array(cell.get_vertex_coordinates()).reshape(3,2)
                area_dict[idx] = tri_area
                mid_vertex_dict[idx,:] = np.array([np.mean(vertex_coords[(i,(i+1)%3),:],axis = 0) for i in range(3)]).reshape(3,2)
         
        mid_vertex_dict = mid_vertex_dict.reshape(-1,2)
        vertx_torch = torch.tensor(mid_vertex_dict).float().to(device)
    else:
        xr_random =  get_interior_points(Nr)
        vertx_torch = xr_random.to(device)
        
    xb = get_boundary_points(Nb)    
    xb_random = xb.detach().cpu().numpy()
    values_b = torch.tensor([u_exp.eval_at_point(x) for x in xb_random]).to(device)

    
    rel_tol = 1
    loss_prev = 1
    epoch = 0
    
    
    t0 = time.time()
    while True:
        
        if not training or (rel_tol <= 1e-5 or epoch == max_epochs+1):
            break
        
        # collocation points are the same as the mesh vertices        
        
        xb = xb.to(device)
        output_b = model(xb)
   
        loss_r = 0
        
        vertx_torch.requires_grad_()
        u_vertices = model(vertx_torch)
        grads = autograd.grad(outputs=u_vertices, inputs=vertx_torch,grad_outputs=torch.ones_like(u_vertices),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
    
        if OT_mesh:
            loss_vertices = torch.sum(torch.square(grads), dim=1).to(device)
            loss_vertices = loss_vertices[::3] + loss_vertices[1::3] + loss_vertices[2::3]
            loss_r = 0.5*torch.dot(area_dict,loss_vertices)/3
        else:
            loss_r = torch.sum(torch.square(grads),dim=1).to(device)
            loss_r = 0.5*torch.mean(loss_r)
        
        loss_b = beta*torch.mean(torch.pow(output_b.squeeze() - values_b, 2))
        loss = loss_r + loss_b

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 1000 == 0:
            print('epoch:', epoch, 'loss:', loss.item(), 'loss_r:',
                  loss_r.item(), 'loss_b:', loss_b.item())

            u.vector()[:] = model(torch.tensor(coords).float().to(device)).detach().cpu().squeeze().numpy()
            L2_err = np.sqrt(assemble((u - u_exp)*(u - u_exp)*dx(mesh_eval))/assemble(u_exp*u_exp*dx(mesh_eval)))        
            if torch.abs(loss).item() < best_loss:
                best_loss = loss
                best_epoch = epoch
                best_err = L2_err
                u_best = u
                torch.save(model.state_dict(), net_model + f'/deep_ritz_eval_OT_{eval_OT}.mdl')
            
            rel_tol = abs(loss - loss_prev)/loss_prev
            loss_prev = loss
            np.savez(var_model + f'/vars_{epoch}.npz', time=time.time() - t0, loss_r=loss_r.item(
            ), loss_b=loss_b.item(), loss=loss.item(), err=L2_err)
        
        epoch  += 1
        
    if save_sol:
        File(net_model + f'/u_eval_OT_{eval_OT}.pvd') << u_best
    
    if training:
        print('best epoch:', best_epoch, 'best loss:', best_loss,'best error',best_err)

    # plot figure
    model.load_state_dict(torch.load(net_model + f'/deep_ritz_eval_OT_{eval_OT}.mdl',map_location=torch.device('cpu')))

    with torch.no_grad():

        x = torch.linspace(-1, 1, 1001)
        y = torch.linspace(-1, 1, 1001)

        X, Y = torch.meshgrid(x, y)
        Z = torch.cat((Y.flatten()[:, None], Y.T.flatten()[:, None]), dim=1)
        Z_f = inside_domain(Z)
        Z_f = Z_f.to(device)
        pred = model(Z_f)
    
    if plt_pred:
        plt.figure()
        pred = pred.cpu().numpy()
        pred = pred.reshape(1001, 1001)
    
        ax = plt.subplot(1, 1, 1)
        plt.imshow(pred, interpolation='nearest', cmap='coolwarm',
                       extent=[-1, 1, -1, 1],
                       origin='lower', aspect='auto')
        
        if OT_mesh:
            plt.scatter(xr_OT[:, 0], xr_OT[:, 1], c='black', marker='o', s=0.5)
            plt.scatter(xb_OT[:,0], xb_OT[:, 1], c='green', marker='x', s=5, alpha=0.7)
        else:
            plt.scatter(xr_random[:, 0], xr_random[:, 1], c='black', marker='o', s=0.5)
            plt.scatter(xb_random[:,0], xb_random[:, 1], c='green', marker='x', s=5, alpha=0.7)
            
            
        #divider = make_axes_locatable(ax)
        #cax = divider.append_axes("right", size="5%", pad=0.05)
        #plt.colorbar(h, cax=cax)
        #plt.savefig(net_model + f'/sol.png')

        plt.plot()
        plt.show()

if __name__ == '__main__':
    #dof_to_Nr = {65:33, 225: 161, 833: 705, 3201:2945}
    #beta_vec = [1,10,100,500,1000]
    beta_vec = [500]
    dof_vec = [833]
    for beta in beta_vec:
        for dof in dof_vec:    
            main(beta,dof)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim, autograd

from dolfin import *
import mshr

import numpy as np
import pandas as pd

import os
import time

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pylab as pylt

parameters['allow_extrapolation'] = True
parameters["form_compiler"]["optimize"] = True  # optimize compiler options
# optimize code when compiled in c++
parameters["form_compiler"]["cpp_optimize"] = True
set_log_active(False)  # handling of log messages, warnings and errors.


# Try to solve the poisson equation:
'''  Solve the following PDE
-\Delta u(x) = 1, x\in \Omega,
u(x) = 0, x\in \partial \Omega
\Omega = (-1,1) * (-1,1) \ [0,1) *{0}
'''

# Expression for exact solution


class Expression_u(UserExpression):

    def __init__(self, omega, **kwargs):
        super().__init__(**kwargs)  # This part is new!
        self.omega = omega

    def eval(self, value, x):

        r = sqrt(x[0]*x[0] + x[1]*x[1])
        theta = np.arctan2(abs(x[1]), abs(x[0]))

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
        theta = np.arctan2(abs(x[1]), abs(x[0]))

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
    OUTPUT:    d_v = vertex_to_dof_map(V)
        y -- size (N,*)
    """

    def __init__(self, inplace=False, power=3):
        super(PowerReLU, self).__init__()
        self.inplace = inplace
        self.power = power

    def forward(self, input):
        y = F.relu(input, inplace=self.inplace)
        return torch.pow(y, self.power)


def swish(x):
    return x*torch.sigmoid(x)


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
        #self.phi = phi
        #self.phi = nn.Sigmoid()

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
            xb.append(z)

        elif(near(z[0], 0) and z[1] <= 0):
            xb.append(z)
        else:
            xr.append(z)

    xr = np.array(xr)
    xb = np.array(xb)

    xr = torch.tensor(xr).float()
    xb = torch.tensor(xb).float()

    return xr, xb


def inside_domain(Z):
    Z_f = Z.clone()

    for i, z in enumerate(Z):
        if (z[0] > 1e-16) & (z[1] < 1e-16):
            Z_f[i] = np.nan

    return Z_f


# 65,225,833,3201,12545,49665,197633,225,833
def main(beta_, dof_):

    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device used is: ', device)

    plt_exact = False
    plt_pred = True
    OT_mesh = True
    eval_OT = False
    save_mesh = False
    save_sol = True
    save_coll_points = True
    training = False

    omega = 3/2*pi
    dof_OT = dof_
    domain_vertices = [Point(0.0, 0.0),
                       Point(1.0, 0.0),
                       Point(1.0, 1.0),
                       Point(-1.0, 1.0),
                       Point(-1.0, -1.0), Point(0.0, -1.0)]

    geometry = mshr.Polygon(domain_vertices)

    max_epochs = 50000
    lr = 1e-4
    in_N = 2
    m = 20
    out_N = 1
    depth = 4
    full_path = os.path.realpath(__file__)
    path, filename = os.path.split(full_path)

    string_mesh = f'mesh_OT/mesh_OT_0.44_{dof_OT}.xml.gz'
    mesh_OT = Mesh(string_mesh)
    
    geometry = mshr.Polygon(domain_vertices)
    df = pd.read_csv('coords_to_res.csv')
    if dof_OT in df['coords'].values:
        res = df[df['coords'] == dof_OT]['res']
    else:
        while dof_OT not in df['coords'].values:
            dof_OT += 1
            res = df[df['coords'] == dof_OT]['res']
    mesh_del = mshr.generate_mesh(geometry, res)

    V = FunctionSpace(mesh_OT, "CG", 1)  # function space for solution u
    u_exp = Expression_u(omega, degree=5)
    coords = V.tabulate_dof_coordinates()
    xr_mesh, xb_mesh = get_interior_boundary_mesh(coords)
    
    if not eval_OT:
        V = FunctionSpace(mesh_del, "CG", 1)  # function space for solution u
        u = Function(V)
        u_best = Function(V)
        coords = V.tabulate_dof_coordinates()
        mesh = mesh_del
        
    Nr = len(xr_mesh)
    Nb = len(xb_mesh)
    
    if OT_mesh:
        xr = xr_mesh
        xb = xb_mesh
    else:
        xr = get_interior_points(Nr)
        xb = get_boundary_points(Nb)

    if plt_exact:
        u = interpolate(u_exp, V)
        pylt.figure(figsize=(10, 10))
        p = plot(u)
        # set colormap
        p.set_cmap("coolwarm")
        # pylt.colorbar(p)
        pylt.xlabel('x')
        pylt.ylabel('y')
        pylt.show()

    var_name = f'/Nets_DGM/dof_{dof_OT}_m_{m}_depth_{depth}_beta_{beta}_lr_{lr}_epochs_{max_epochs}/Var/vars_coll_{OT_mesh}'
    net_name = f'/Nets_DGM/dof_{dof_OT}_m_{m}_depth_{depth}_beta_{beta}_lr_{lr}_epochs_{max_epochs}/Net_coll_{OT_mesh}'
    var_model = path + var_name
    net_model = path + net_name
    os.makedirs(var_model, exist_ok=True)
    os.makedirs(net_model, exist_ok=True)

    if save_mesh:
        File(net_model + f'/Mesh/mesh_{OT_mesh}.pvd') << mesh

    model = drrnn(in_N, m, out_N, depth).to(device)
    model.apply(weights_init)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_epoch = 0
    best_loss = 1e5

    t0 = time.time()
    XB = xb.detach().cpu().numpy()
    values = torch.tensor([u_exp.eval_at_point(x) for x in XB]).to(device)
    rel_tol = 1
    loss_prev = 1
    epoch = 0

    while True:

        if not training or (rel_tol <= 1e-5 or epoch == max_epochs+1):
            break

        # Sample random points at each iteration
        #xb = get_boundary_points(N=Nb)
        #xr = get_interior_points(N=Nr)

        # save collocation of points once
        if epoch == 2 and save_coll_points:
            np.save(net_model + f'/collocation_points.npy',
                    xr.detach().cpu().numpy())
            np.save(net_model + f'/boundary_points.npy',
                    xb.detach().cpu().numpy())

        xr = xr.to(device)
        xb = xb.to(device)

        XB = xb.detach().cpu().numpy()
        values = torch.tensor([u_exp.eval_at_point(x) for x in XB]).to(device)
        output_b = model(xb)

        # loss function for the inside of the domain
        xr.requires_grad_()
        output_r = model(xr)

        grad = autograd.grad(outputs=output_r, inputs=xr, grad_outputs=torch.ones_like(output_r),
                             create_graph=True, retain_graph=True, only_inputs=True)[0]

        laplacian = autograd.grad(outputs=grad, inputs=xr, grad_outputs=torch.ones_like(grad),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        # array to store values of laplacian
        loss_r = torch.sum(torch.square(laplacian), dim=1)
        loss_r = torch.mean(loss_r)
        loss_b = beta*torch.mean(torch.square(output_b.squeeze() - values))
        loss = loss_r + loss_b

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 1000 == 0:

            u.vector()[:] = model(torch.tensor(
                coords).float().to(device)).detach().cpu().squeeze().numpy()

            L2_err = np.sqrt(assemble((u - u_exp)*(u - u_exp)
                             * dx(mesh))/assemble(u_exp*u_exp*dx(mesh)))

            if loss.item() < best_loss:
                best_loss = loss
                best_epoch = epoch
                best_err = L2_err
                torch.save(model.state_dict(), net_model +
                           f'/deep_ritz.mdl')

            rel_tol = abs(loss - loss_prev)/loss_prev
            loss_prev = loss
            np.savez(var_model + f'/vars_{epoch}.npz', time=time.time() - t0, loss_r=loss_r.item(
            ), loss_b=loss_b.item(), loss=loss_r.item() + loss_b.item(), err=L2_err)

            print('epoch:', epoch, 'loss:', loss.item(), 'loss_r:',
                  loss_r.item(), 'loss_b:', loss_b.item())

        epoch += 1

    #print('best epoch:', best_epoch, 'best loss:',
    #      best_loss, 'L2_err', best_err)
   

    # plot figure
    if plt_pred:
        model.load_state_dict(torch.load(net_model + f'/deep_ritz.mdl'))
        
        u_best.vector()[:] = model(torch.tensor(
                coords).float().to(device)).detach().cpu().squeeze().numpy()
        if save_sol:
            File(net_model + f'/u.pvd') << u_best
        with torch.no_grad():

            x = torch.linspace(-1, 1, 1001)
            y = torch.linspace(-1, 1, 1001)

            X, Y = torch.meshgrid(x, y)
            Z = torch.cat(
                (Y.flatten()[:, None], Y.T.flatten()[:, None]), dim=1)
            Z_f = inside_domain(Z)
            Z_f = Z_f.to(device)
            pred = model(Z_f)

        plt.figure()
        pred = pred.cpu().numpy()
        pred = pred.reshape(1001, 1001)

        ax = plt.subplot(1, 1, 1)
        plt.imshow(pred, interpolation='nearest', cmap='coolwarm',
                   extent=[-1, 1, -1, 1],
                   origin='lower', aspect='auto')
        plt.scatter(xr[:, 0].detach().cpu().numpy(), xr[:, 1].detach(
        ).cpu().numpy(), c='black', marker='o', s=0.5)
        plt.scatter(xb[:, 0].detach().cpu().numpy(), xb[:, 1].detach(
        ).cpu().numpy(), c='green', marker='x', s=5, alpha=0.7)

        #divider = make_axes_locatable(ax)
        #cax = divider.append_axes("right", size="5%", pad=0.05)
        #plt.colorbar(h, cax=cax)
        plt.savefig(net_model + f'/sol.png')
        # plt.show()


if __name__ == '__main__':
    #dof_to_Nr = {65:33, 225: 161, 833: 705, 3201:2945}
    #beta_vec = [1,10,100,500,1000]
    beta_vec = [1000]
    dof_vec = [833]
    for beta in beta_vec:
        for dof in dof_vec:
            main(beta, dof)

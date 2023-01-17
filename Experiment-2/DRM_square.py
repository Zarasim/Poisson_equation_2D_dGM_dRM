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


import matplotlib.pyplot as plt
import pylab as pylt

parameters['allow_extrapolation'] = True
parameters["form_compiler"]["optimize"] = True  # optimize compiler options
# optimize code when compiled in c++
parameters["form_compiler"]["cpp_optimize"] = True
set_log_active(False)  # handling of log messages, warnings and errors.


# Expression of exact solution
class Expression_u(UserExpression):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # This part is new!

    def eval(self, value, x):

        value[0] = sin(pi*x[0])*sin(pi*x[1])

    def eval_at_point(self, x):

        value = sin(pi*x[0])*sin(pi*x[1])

        return value

    def value_shape(self):
        return ()


class PowerReLU(nn.Module):
    """
    Implements simga(x)^(power)
    Applies a power of the rectified linear unit element-wise.

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
    randomly sample N points from interior of [0,1]^d
    """
    # return points for each block
    x = torch.rand(N).unsqueeze(-1)
    y = torch.rand(N).unsqueeze(-1)
    X = torch.cat((x, y), dim=1)

    return X


def get_boundary_points(N=33):
    index = torch.rand(N, 1)             # [0,1]

    # x in (0,1) y = 0
    xb1 = torch.cat((index, torch.zeros_like(index)), dim=1)

    # x = 1 y in (0,1)
    xb2 = torch.cat((torch.ones_like(index), index), dim=1)

    # x in (0,1) y = 1
    xb3 = torch.cat((index, torch.ones_like(index)), dim=1)

    # x = 0 y in (0,1)
    xb4 = torch.cat((torch.zeros_like(index), index), dim=1)

    xb = torch.cat((xb1, xb2, xb3, xb4), dim=0)

    return xb


def get_interior_boundary_mesh(coords):

    xr = []
    xb = []

    for z in coords:
        if z[0] >= 0.99 or z[1] >= 0.99 or z[0] <= 0.01 or z[1] <= 0.01:
            xb.append(z)
        else:
            xr.append(z)

    xr = np.array(xr)
    xb = np.array(xb)

    xr = torch.tensor(xr).float()
    xb = torch.tensor(xb).float()

    return xr, xb


def main(beta_, dof_):

    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device used is: ', device)

    plt_exact = 0
    plt_pred = 1
    square_mesh = True
    eval_square = False  # evaluate loss on regularly spaced points
    save_mesh = False
    save_sol = False
    training = 1
    domain_vertices = [Point(1.0, 0.0),
                       Point(1.0, 1.0),
                       Point(0, 1.0),
                       Point(0, 0)]

    dof_square = dof_
    beta = beta_
    torch_pi = torch.acos(torch.zeros(1)).item() * 2  # 3.1415927410125732

    max_epochs = 50000
    lr = 1e-4
    in_N = 2
    m = 20
    out_N = 1
    depth = 4

    mesh = RectangleMesh(Point(0, 0), Point(
        1, 1), dof_square, dof_square, 'crossed')

    geometry = mshr.Polygon(domain_vertices)
    df = pd.read_csv('coords_to_res.csv')
    if dof_square in df['coords'].values:
        res = df[df['coords'] == dof_square]['res']
    else:
        while dof_square not in df['coords'].values:
            dof_square += 1
            res = df[df['coords'] == dof_square]['res']

    mesh_del = mshr.generate_mesh(geometry, res)

    if eval_square:
        mesh_eval = Mesh(mesh)
    else:
        mesh_eval = Mesh(mesh_del)

    V = FunctionSpace(mesh_eval, "CG", 1)
    u = Function(V)
    u_best = Function(V)
    coords = V.tabulate_dof_coordinates()

    u_exp = Expression_u(degree=5)

    xr_mesh, xb_mesh = get_interior_boundary_mesh(coords)
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
    path, _ = os.path.split(full_path)
    var_name = f'/Nets_DRM_square/dof_{dof_square}_m_{m}_depth_{depth}_beta_{beta}_lr_{lr}_epochs_{max_epochs}/Var/vars_coll_{square_mesh}'
    net_name = f'/Nets_DRM_square/dof_{dof_square}_m_{m}_depth_{depth}_beta_{beta}_lr_{lr}_epochs_{max_epochs}/Net_coll_{square_mesh}'
    var_model = path + var_name
    net_model = path + net_name
    os.makedirs(var_model, exist_ok=True)
    os.makedirs(net_model, exist_ok=True)

    if save_mesh:
        File(net_model + '/Mesh/mesh_square.pvd') << mesh
        File(net_model + '/Mesh/mesh_Delaunay.pvd') << mesh_eval

    model = drrnn(in_N, m, out_N, depth=4).to(device)
    model.apply(weights_init)
    optimizer = optim.Adam(model.parameters(), lr)

    best_epoch = 0
    best_loss = 1e5

    if square_mesh:
        V = FunctionSpace(mesh, "CG", 1)
        ncells = mesh.num_cells()
        area_dict = torch.zeros(ncells).to(device)
        mid_vertex_dict = np.zeros((ncells, 3, 2))

        for cell in cells(mesh):
            idx = cell.index()
            tri_area = cell.volume()
            vertex_coords = np.array(
                cell.get_vertex_coordinates()).reshape(3, 2)
            area_dict[idx] = tri_area
            mid_vertex_dict[idx, :] = np.array(
                [np.mean(vertex_coords[(i, (i+1) % 3), :], axis=0) for i in range(3)]).reshape(3, 2)

        mid_vertex_dict = mid_vertex_dict.reshape(-1, 2)
        vertx_torch = torch.tensor(mid_vertex_dict).float().to(device)
    else:
        xr_random = get_interior_points(Nr)
        vertx_torch = xr_random.to(device)

    loss_r = 0

    rel_tol = 1
    loss_prev = 1
    epoch = 0
    t0 = time.time()

    while True:

        if not training or (rel_tol <= 1e-5 or epoch == max_epochs+1):
            break

        vertx_torch.requires_grad_()

        xb = get_boundary_points(Nb)
        xb = xb.to(device)
        output_b = model(xb)

        u_vertices = model(vertx_torch).to(device).squeeze(-1)
        f_vertices = 2*torch_pi**2 * \
            torch.sin(torch_pi*vertx_torch[:, 0]) * \
            torch.sin(torch_pi*vertx_torch[:, 1]).to(device)

        grads = autograd.grad(outputs=u_vertices, inputs=vertx_torch, grad_outputs=torch.ones_like(u_vertices),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

        if square_mesh:
            loss_vertices = torch.sum(torch.square(grads), dim=1).to(device)
            loss_vertices = loss_vertices[::3] + \
                loss_vertices[1::3] + loss_vertices[2::3]
            fu_vertices = f_vertices[::3] * \
                u_vertices[::3] + f_vertices[1::3]*u_vertices[1::3] + \
                f_vertices[2::3]*u_vertices[2::3]
            loss_r = 0.5*(torch.dot(area_dict, loss_vertices) /
                          3) + torch.dot(area_dict, fu_vertices)/3
        else:
            loss_r = torch.sum(torch.square(grads), dim=1).to(device)
            loss_r = 0.5*torch.mean(loss_r) - torch.mean(f_vertices*u_vertices)

        loss_b = beta*torch.mean(torch.pow(output_b.squeeze(), 2))
        loss = loss_r + loss_b
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 1000 == 0:
            print('epoch:', epoch, 'loss:', loss.item(), 'loss_r:',
                  loss_r.item(), 'loss_b:', loss_b.item())

            u.vector()[:] = model(torch.tensor(coords).float().to(
                device)).detach().cpu().squeeze().numpy()
            L2_err = np.sqrt(assemble((u - u_exp)*(u - u_exp) *
                             dx(mesh_eval)))/assemble(u_exp*u_exp*dx(mesh_eval))

            best_err = L2_err
            if torch.abs(loss).item() < best_loss:
                best_loss = loss
                best_epoch = epoch
                best_err = L2_err
                u_best = u
                torch.save(model.state_dict(), net_model +
                           f'/deep_ritz_eval_square_{eval_square}.mdl')

            rel_tol = abs(abs(loss) - abs(loss_prev))/abs(loss_prev)
            loss_prev = loss
            np.savez(var_model + f'/vars_{epoch}.npz', time=time.time() - t0, loss_r=loss_r.item(
            ), loss_b=loss_b.item(), loss=loss.item(), err=L2_err)

        epoch += 1

    if save_sol:
        File(net_model + f'/u_eval_square_{eval_square}.pvd') << u_best

    if training:
        print('best epoch:', best_epoch, 'best loss:',
              best_loss, 'best error', best_err)

    # plot numerical solution
    if plt_pred:
        model.load_state_dict(torch.load(
            net_model + f'/deep_ritz_eval_square_{eval_square}.mdl', map_location=torch.device('cpu')))

        with torch.no_grad():

            x = torch.linspace(0, 1, 1001)
            y = torch.linspace(0, 1, 1001)
            X, Y = torch.meshgrid(x, y)
            Z = torch.cat(
                (Y.flatten()[:, None], Y.T.flatten()[:, None]), dim=1)
            pred = model(Z)
            plt.figure()
            pred = pred.cpu().numpy()
            pred = pred.reshape(1001, 1001)

        plt.imshow(pred, interpolation='nearest', cmap='coolwarm',
                   extent=[0, 1, 0, 1],
                   origin='lower', aspect='auto')

        if square_mesh:
            plt.scatter(xr_mesh[:, 0], xr_mesh[:, 1],
                        c='black', marker='o', s=0.5)
            plt.scatter(xb[:, 0], xb[:, 1], c='green',
                        marker='x', s=5, alpha=0.7)
        else:
            plt.scatter(xr_random.detach().numpy()[:, 0], xr_random.detach().numpy()[:, 1],
                        c='black', marker='o', s=0.5)
            plt.scatter(xb[:, 0], xb[:, 1],
                        c='green', marker='x', s=5, alpha=0.7)

        plt.plot()
        plt.show()

    return best_err


if __name__ == '__main__':

    beta_vec = [1, 10, 100, 500, 1000]

    # (nx + 1)(ny + 1)
    # 64, 225, 784, 3136, 12544
    dof_vec = [7, 14, 27, 55, 111]

    for beta in beta_vec:
        Err_vec = []
        for dof in dof_vec:
            Err = main(beta, dof)
            Err_vec.append(Err)
        dict_Err = {'dof': dof_vec, 'err': Err_vec}
        df = pd.DataFrame(Err_vec)
        df.to_csv(f'Error_square_beta_{beta}.csv', index=False)

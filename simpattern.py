#!/usr/bin/env python

#######################################################################################
# AUTHOR: Tanmoy Sanyal, Shell group, Chemical Engineering Department, UC Santa Barbara
# USAGE: python simpattern.py m n k save_location
# m, n --> integer (currently only the following values supported: (0,1), (1,1), (3,3)
# k --> float
# save_location --> string
#######################################################################################


import numpy as np
import os, sys
from scipy.special import jv as besselj
from femlib import fem, pde
import matplotlib.pyplot as  plt

# calculate neutral stability boundary
def calcNeutralStability(rate_const = None, c_10 = 10., c_20 = 0.05, eigdict = None, besselmode = None):
        if eigdict is None:
                eigdict = {(0,1): 3.8317 , (1,1): 1.8412, (3,3): 11.3459} # from Wolfram alpha
        
        k_mn = eigdict[besselmode]
        slope = ( -c_20 * (c_20 - 2*c_10) ) / k_mn**2.
        D = rate_const * slope
        return (k_mn, D) 


data_dir = './data'
if not os.path.isdir(data_dir): os.mkdir(data_dir)
initmeshfile = os.path.join(data_dir, 'initmesh.mat')  # if this file does not exist, please run the test bench to generate it

# system parameters (make sure that k and D conform to the correct region of the neutral stability line)
c_10 = 1.
c_20 = 0.05
m = int(sys.argv[1])
n = int(sys.argv[2])
k = float(sys.argv[3])
simdata_dir = sys.argv[4]
k_mn, D = calcNeutralStability(c_10 = c_10, c_20 = c_20, rate_const = k, besselmode = (m,n))
D /= 10.

# set up data locations
if not os.path.isdir(simdata_dir): os.mkdir(simdata_dir)
outfile_fmt = os.path.join(simdata_dir, 'pattern%d.dat')

# init mesh
K0 = fem.importInitMesh(matfile = initmeshfile)
mesh = fem.Mesh(K0)
        
# set zero Neumann boundary conditions
mesh.NeumannEdges = mesh.BoundaryEdges
mesh.NumNeumannEdges = len(mesh.NeumannEdges)
def gNeumann(Mesh = mesh): return [lambda p: 0.]*2
        
# get boundary partition into Dirichlet nodes and free nodes
mesh.partitionNodes()
        
# assemble stiffness and mass matrices on the mesh
a = fem.Assemb(Mesh = mesh)
a.AssembStiffMat()
a.AssembMassMat()
W = a.globalStiffMat
M = a.globalMassMat
        
# set source function
def fsrc(Mesh = mesh):  
        return [lambda p,mesh,u: -k*u[0]*u[1]**2.,
                lambda p,mesh,u:  k*u[0]*u[1]**2. ]
                               
# assemble LHS of final lin. alg problem
def makeBlockMat(x): 
        zero = np.zeros([x.shape[0], x.shape[1]])
        blockmat = np.bmat([ [D*x, zero], [zero, D*x] ])
        return np.array(blockmat)
        
def getlhs(W,M): return M


# define the PDE
pde = pde.Parabolic(NComponents = 2, Mesh = mesh, StiffMat = a.globalStiffMat, MassMat = a.globalMassMat)
pde.setNeumannFunc = gNeumann
pde.setSrcFunc = fsrc
pde.AssembBlockStiffMat = makeBlockMat
pde.AssembBlockMassMat = makeBlockMat
pde.AssembLHS = getlhs
import linsolv
W_free = makeBlockMat(pde.getFreeNodeArray(W))

# initial condition (bessel perturbation)
#J_m (k_mn * r) * Cos(theta) written in cartesian co-ordinates
f0 = [lambda p: c_10 + 0.03*c_10 * besselj(m, k_mn * np.sqrt(p[0]**2 + p[1]**2)) * p[0]/(1e-4 + np.sqrt(p[0]**2 + p[1]**2)),
      lambda p: c_20 + 0.03*c_20 * besselj(m, k_mn * np.sqrt(p[0]**2 + p[1]**2)) * p[0]/(1e-4 + np.sqrt(p[0]**2 + p[1]**2))]
        
u0 = np.zeros([mesh.NumNodes,2])
for i, node in enumerate(mesh.Nodes): 
        u0[i,0] = f0[0](node)
        u0[i,1] = f0[1](node)
        
# start time loop 
np.savetxt(outfile_fmt % 0, u0)
dt = 1e-4
        
print "dt = ", dt 
print "Mesh size = ", mesh.Diam
raw_input('Press any key to start time loop')
        
NSteps = 200
u_old = u0
err = []
loop = True # turn this off if you already have the data and just want to plot
if loop:
        for i in range(NSteps):
                print 'Timestep: ', i
                pde.u = u_old
                M_free, vecs = pde.AssembPDE()
                F = vecs[0]; G = vecs[1]
                x_old = pde.wrapSol(u_old)
                rhs = linsolv.mul(a = M_free, b = x_old) - dt*linsolv.mul(a = W_free, b = x_old) + dt*F + dt*G 
                x_new = pde.Solve(M_free, rhs)
                u_new = pde.unwrapSol(x_new)
                        
                # write to file and update
                np.savetxt(outfile_fmt%(i+1), u_new)     
                err.append(abs(u_new[:,0] - u_old[:,0]).max())
                u_old = u_new
                  
# pattern animations
print '\nTime stepping ended...'
raw_input('Press any key to start animation...')
p = fem.Plot(Mesh = mesh)
fig = plt.figure(figsize = (7,5), facecolor = 'w', edgecolor = 'w')
datafilelist = []
[datafilelist.append(outfile_fmt % x) for x in range(NSteps)]
ax = fig.add_subplot(1, 1, 1)
p.ax = ax
p.patternAnimate(dataFileList = datafilelist, Component = 0, delay = 0.01,
                 frameRate = 10, Prefix = os.path.join(simdata_dir, 'pattern%d%d' % (m,n)))

# plot convergence
fig_err = plt.figure(figsize = (7,5), facecolor = 'w', edgecolor = 'w')
ax_err = fig_err.add_subplot(1,1,1)
ax_err.set_xlabel('Timestep', fontsize = 'xx-large'); ax_err.set_ylabel('Convergence to steady state', fontsize = 'xx-large')
ax_err.plot(err)


plt.show()

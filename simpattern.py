#!/usr/bin/env python

import os
import numpy as np
from femlib import fem, solver

# compile background fortran 
solver.recompile = True
solver.fort_compile()
import bkeuler

# problem parameters
D1 = 1. 
D2 = 1.
#f = 
#g = 
delta_t = 0.1 
NTime = 10 

# data locations
initmeshfile = './testdata/testmesh0.mat'
data_dir = './testdata/simpattern_test'
outfile_fmt = os.path.join(data_dir, 'pattern%d.dat')
if not os.path.isdir(data_dir): os.mkdir(data_dir)

# LAPACK lib path
LAPACK_PATH = '/usr/lib/lapack'

# generate mesh and shapefunction space
mesh0 = fem.importInitMesh(matfile = initmeshfile)
K = fem.Mesh(mesh0)
P = fem.ShapeFn(Mesh = K, Element = None)

# setup initial condition on mesh
u1_init = lambda x,y : x**2. + y**2. 
u2_init = lambda x,y : 1.0
U10 = np.zeros([K.NumNodes,1])
U20 = np.zeros([K.NumNodes,1])
for i in range(K.NumNodes):
        U10[i] = u1_init(K.Nodes[i,0], K.Nodes[i,1])
        U20[i] = u2_init(K.Nodes[i,0], K.Nodes[i,1])
U0 = np.array([U10, U20]).flatten().reshape(2*K.NumNodes,1)

# assemble global stiffness and mass matrices on the mesh
a = fem.Assemb(Mesh = K)
a.AssembMat_naive()
W = a.globalStiffMat
M = a.globalMassMat
Areas = a.getAllElementAreas()

# block matrices for 2 component system
zero = np.zeros([K.NumNodes, K.NumNodes])
W2 = np.array(np.bmat([[D1*W, zero], [zero, D2*W]]))
M2 = np.array(np.bmat([[M, zero], [zero, M]]))

# assembling the initial right hand side (directly for 2 component system)
F20 = np.zeros([2*K.NumNodes,1])
F20 = bkeuler.assembf(f = F20, u = U0, elements = K.Elements, areas = Areas)

# initiate solver
solver.K = W2 
solver.M = M2
solver.Elements = K.Elements
solver.ElemAreas = Areas
solver.Delta_t = delta_t

# start the time loop
U = U0
F = F20
#TODO: check why the solver is not entering the subroutine bkeuler()
for i in range(NTime):
        print 'Time Iteration: ', i
        solver.U0 = U
        solver.F0 = F20
        U,F = solver.bkEuler()
        solver.U0 = U
        solver.F0 = F
        # log data
        np.savetxt(outfile_fmt % i, U)
        


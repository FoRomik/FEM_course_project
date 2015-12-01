#!/usr/bin/env python

import numpy as np
from femlib import fem, solver

# problem parameters
#D1 = 
#D2 = 
#f = 
#g = 
#delta_t = 
#NTime = 

# data locations
#initmeshfile =
#data_dir = 
#outfile_fmt = 'pattern%d.dat'

# LAPACK lib path
LAPACK_PATH = '/usr/lib/lapack'

# generate mesh and shapefunction space
mesh0 = fem.importInitMesh(matfile = initmeshfile)
K = fem.Mesh(mesh0)
P = fem.ShapeFn(Mesh = K, Element = None)

# setup initial condition on mesh
#u1_init = lambda x,y :: 
#u2_init = lambda x,y ::
U10 = np.zeros([K.NumNodes,1])
U20 = np.zeros([K.NumNodes,1])
for i in range(K.NumNodes):
        U10[i] = u1_init(K.Nodes[i])
        U20[i] = u2_init(K.Nodes[i])
U0 = np.array([U10, U20]).flatten().reshape(2*K.NumNodes,1)

# assemble global stiffness and mass matrices on the mesh
a = fem.Assemb(Mesh = K, f = None)
a.AssembMat_naive()
W = a.globalStiffMat
M = a.globalMassMat
Areas = a.getAllElementAreas()

# block matrices for 2 component system
zero = np.zeros([K.NumNodes, K.NumNodes])
W2 = np.array(np.bmat([[D1*W, zero], [zero, D2*W]])
M2 = np.array(np.bmat([[M, zero], [zero, M]])

# assembling the initial right hand side (directly for 2 component system)
solver.fcompile()
F20 = np.zeros([2*K.NumNodes,1])
F20 = solver.assembf(f = F, u = U0, Elements = K.Elements, Areas = Areas)

# initiate solver
solver.fcompile(recompile = True)
solver.K = W2 
solver.M = M2
solver.ElemAreas = Areas
solver.Delta_t = delta_t

# start the time loop
U = U0
F = F20
for i in range(NTime):
        solver.U0 = U
        solver.F0 = F20
        U,F = solver.bkEuler()
        solver.U0 = U
        solver.F0 = F
        # log data
        np.savetxt(outfile_fmt % i, U)
        


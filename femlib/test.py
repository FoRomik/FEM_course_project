#!/usr/bin/env python


#######################################################################################
# AUTHOR: Tanmoy Sanyal, Shell group, Chemical Engineering Department, UC Santa Barbara
# test bench for fem and pde modules
#######################################################################################


import os

import numpy as np
import scipy as scp
from scipy import stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as tri

from fem import *
from pde import *

MATLAB_EXEC = 'matlab'
testdata_dir = '../testdata'
if not os.path.isdir(testdata_dir): os.mkdir(testdata_dir)


# before using this test bench the matlab script genMesh.m should be run to produce the testinitmesh.mat which 
# is used as the initial mesh for most of the tests here. It can be called from the command line though
initmeshfile = os.path.join(testdata_dir, 'testinitmesh.mat')
if not os.path.isfile(initmeshfile):
        cmdstring = '''%s -nojvm -nodisplay -nosplash -r "genMesh(%g, '%s'); exit" ''' % (MATLAB_EXEC, 0.1, initmeshfile)
        os.system(cmdstring)
        
#=============================================================================================================================================================
def testMesh():
        K0 = importInitMesh(matfile = initmeshfile)
        m = Mesh(K0)
        
        m.NeumannEdges = m.BoundaryEdges 
        m.NeumannEdges = np.delete(m.NeumannEdges, 0, 0)
        m.NeumannEdges = np.delete(m.NeumannEdges, 14, 0)
        m.NumNeumannEdges = len(m.NeumannEdges)
       
        print m.NumNodes
        print m.NumElements
        
        m.DirEdges = np.array([[0,8], [0,15]])
        m.NumDirEdges = len(m.DirEdges)
        
	m.partitionNodes()
	print m.DirNodes
	print m.FreeNodes
	
	fig = plt.figure(figsize = (7,5), facecolor = 'w', edgecolor = 'w')
	ax1 = fig.add_subplot(121); ax2 = fig.add_subplot(122)
	p = Plot(Mesh = m)
	p.ax = ax1; p.plotBoundaries()
	
	m.refineMesh()
	m.partitionNodes()
	print m.DirNodes
	print m.FreeNodes
	p.ax = ax2; p.plotBoundaries()

#=============================================================================================================================================================
def testShapeFnPlot():	
        K0 = importInitMesh(matfile = initmeshfile)
        m = Mesh(K0); m.NeumannEdges = m.BoundaryEdges ;  m.NumNeumannEdges = len(m.NeumannEdges)
	fig = plt.figure(figsize = (7,5), facecolor = 'w', edgecolor = 'w')
	ax3d = Axes3D(fig)
	Node = m.Nodes[50]
	p = Plot(Mesh = m, ax = ax3d)
	p.plotShapeFunc(Node)  

#=============================================================================================================================================================
def testPatternPlot():
	K0 = importInitMesh(matfile = initmeshfile)
        m = Mesh(K0); m.NeumannEdges = m.BoundaryEdges ;  m.NumNeumannEdges = len(m.NeumannEdges)
	m.refineMesh();m.refineMesh()
	fig = plt.figure(figsize = (7,5), facecolor = 'w', edgecolor = 'w')
	ax = fig.add_subplot(111)
	u = np.zeros(m.NumNodes)
	for i in range(m.NumNodes):
		u[i] = np.cos(m.Nodes[i,0] + m.Nodes[i,1])
	p = Plot(Mesh = m, ax = ax)
	p.patternPlot(u = u)

#=============================================================================================================================================================
def testElementMat():
	K0 = importInitMesh(matfile =initmeshfile)
        m = Mesh(K0); m.NeumannEdges = m.BoundaryEdges ;  m.NumNeumannEdges = len(m.NeumannEdges)
	T = m.Elements[22]	
	s = ShapeFn(Mesh = m, Element = T)
	stiffmat = s.StiffMatElement_P1()
	massmat = s.MassMatElement_P1()		
	phi = []
	[phi.append(s.getLocalShapeFn(node)) for node in m.Nodes[T]]	

	print 'Element =', T
	print 'Nodes = ', m.Nodes[T]
	print 'phi = ', phi
	print 'stiffmat = ', stiffmat
	print 'massmat = ', massmat

#=============================================================================================================================================================
def testAssembly():
	K0 = importInitMesh(matfile = initmeshfile)
        m = Mesh(K0)
	
	m.NeumannEdges = m.BoundaryEdges 
        m.NeumannEdges = np.delete(m.NeumannEdges, 0, 0)
        m.NeumannEdges = np.delete(m.NeumannEdges, 14, 0)
        m.NumNeumannEdges = len(m.NeumannEdges)
       
        m.DirEdges = np.array([[0,8], [0,15]])
        m.NumDirEdges = len(m.DirEdges)
        
        m.partitionNodes()
        
        print '\n\n================CHECK MESH NODE SETUP ===================\n\n'
        print "NumDirNodes = ", m.NumDirNodes
	print "NumFreenodes = ", m.NumFreeNodes
	print "DirNodes = ", m.DirNodes
	print "FreeNodes = ", m.FreeNodes
	
	a = Assemb(Mesh = m)
	a.AssembStiffMat()
	a.AssembMassMat()
	
	print '\n\n================CHECK MATRICES ASSEMBLED OVER MESH===================\n\n'
	print "GlobalStiffMat =", a.globalStiffMat
	print "GlobalMassMat =", a.globalMassMat
	print "StiffMatsize = ", a.globalStiffMat.shape
	print "MassMatSize = ", a.globalMassMat.shape
       
        pde = Elliptic(Mesh = m, StiffMat = a.globalStiffMat, MassMat = a.globalMassMat)
        
        # set source term and boundary conditions
        def fsrc(Mesh = m): return [lambda p: 10 * p[0] * p[1]]
        def gdir(Mesh = m): return [lambda p: 1]
        def gneumann(Mesh = m): return [lambda p: 0]
        pde.setSrcFunc = fsrc     
        pde.setDirFunc = gdir
        pde.setNeumannFunc = gneumann     
        
        allsrc = pde.getAllSrc()
        srcterm = pde.getSrcTerm()
        neumannbc = pde.getNeumannBC()
        
        print '\n\n================CHECK ASSEMBLED RHS VECTOR===================\n\n'
        print "AllSrc = ", allsrc
        print "Src Term = ", srcterm
        print "NeumannBC = ", neumannbc
        print "SrcTermSize = ", srcterm.shape
        print "NeumannBCSize = ", neumannbc.shape
        
        def makeBlock(X): return X
        def getLHS(K,M): return K+M
        pde.AssembBlockStiffMat = makeBlock
        pde.AssembBlockMassMat = makeBlock 
        pde.AssembLHS = getLHS
        dirbc = pde.getDirBC()
        A, b = pde.AssembPDE()
        
        print '\n\n================CHECK ASSEMBLED LHS MATRIX===================\n\n'
        print "DirBC = ", dirbc
        print "A = ", A
        print "b =", b[0]+b[1]-b[2]
        print "DirBCSize = ", dirbc.shape
        print "ASize = ", A.shape
        print "bsize =", b[0].shape, b[1].shape, b[2].shape
        
#=============================================================================================================================================================        
def testLinSolver():
        K0 = importInitMesh(matfile = initmeshfile)
        m = Mesh(K0)
        sol = PDE(Mesh = m)
        A = np.random.random((10,10))
        b = np.random.random((10,1))
        fort_sol = sol.Solve(A,b)
        numpy_sol = np.linalg.solve(A,b)
        print 'Solving the system .. Ax = b'
        print 'A = \t', A
        print 'b= \t', b
        print 'Fortran DGSEV solution \t ', fort_sol
        print '\n\n----------------------\n\n'
        print 'Numpy solution \t ', numpy_sol
        
#=============================================================================================================================================================         
def testPoisson(meshmatfile = initmeshfile, showPlot = True):
        # init mesh
        K0 = importInitMesh(matfile = meshmatfile)
        m = Mesh(K0)
        
        # elliptic test problem: u(x,y) = 1 -x - y
        # for other test functions, please change the lambda function f 
        # and the lambda function in fsrc
        
        NComponents = 1
        f = [lambda p: (1 - p[0] - p[1])/1.]
             
        # set boundary conditions
        m.DirEdges = m.BoundaryEdges
        m.NumDirEdges = len(m.DirEdges)
        def gdir(Mesh = m): return f
        
        # get boundary partition
        m.partitionNodes()
        
        # assemble stiffness and mass matrices
        a = Assemb(Mesh = m)
	a.AssembStiffMat()
	a.AssembMassMat()
        K = a.globalStiffMat
        M = a.globalMassMat
        
        # set source function
        def fsrc(Mesh = m):  return [lambda p: 0.]
                               
        # assemble LHS of final lin. alg problem
        def makeBlockMat(x): return x
        def getlhs(K,M): return K
        
        # define the PDE
        pde = Elliptic(NComponents = NComponents, Mesh = m, StiffMat = a.globalStiffMat, MassMat = a.globalMassMat)
        pde.setDirFunc = gdir
        pde.setSrcFunc = fsrc
        pde.AssembBlockStiffMat = makeBlockMat
        pde.AssembBlockMassMat = makeBlockMat
        pde.AssembLHS = getlhs
        
        # generate final lin alg problem
        print 'Assembling Lin. Alg. problem...'
        A, rhs = pde.AssembPDE()
        b = rhs[0] + rhs[1] - rhs[2]
        
        # solve the lin alg problem naively
        x = pde.Solve(A,b)
        
        # construct the full FEM solution 
        u = pde.unwrapSol(x)

        # construc the full initial solution
        u_ex = np.zeros([m.NumNodes, 1])
        for i in range(m.NumNodes): u_ex[i,0] = f[0](m.Nodes[i])
        
        # plot the solution and compare with analytical sol
        # general plot script for N components
        if showPlot:
                p = Plot(Mesh = m)
	        fig = plt.figure(figsize = (16,4), facecolor = 'w', edgecolor = 'w')
        	ax1 = fig.add_subplot(1, 3, 1)
        	ax2 = fig.add_subplot(1, 3, 2)
        	ax3 = fig.add_subplot(1, 3, 3)
                ax1.set_title(r'$u_{exact}$', fontsize = 'xx-large', fontweight = 'bold')
        	ax2.set_title(r'$u_{FEM}$', fontsize = 'xx-large', fontweight = 'bold')
        	ax3.set_title(r'$u_{exact} - u_{FEM}$', fontsize = 'xx-large', fontweight = 'bold')
        	p.ax = ax1; p.patternPlot(u_ex)
        	p.ax = ax2; p.patternPlot(u)
        	p.ax = ax3; p.patternPlot(u_ex - u)
	
	                
	# calculate error norm
	L2err  = np.linalg.norm(u -u_ex)
	print 'L2 error = ', L2err

	return L2err
	
#=============================================================================================================================================================
def testPoissonErrorScaling():
	meshmatfile_fmt = 'testmesh%d.mat'
	hmax = np.array([0.04, 0.06, 0.08, 0.1, 0.4, 0.5, 0.8])
	N = len(hmax)
	err = np.zeros(N)
	for i in range(N):
	        h = hmax[i]
	        print '\n\nGenerating mesh of size  = ', h
	        meshmatfile = os.path.join(testdata_dir, meshmatfile_fmt % i)
	        if not os.path.isfile(meshmatfile):
	                cmdstring = '''%s -nojvm -nodisplay -nosplash -r "genMesh(%g, '%s'); exit" ''' % (MATLAB_EXEC, h, meshmatfile)
        	        os.system(cmdstring)
	        print 'Solving Poisson problem on mesh of size = ', h
	        err[i] = testPoisson(meshmatfile = meshmatfile, showPlot = False)
	      
	# fit straight line to error data
	print err ; raw_input()
	logh = np.log10(1./hmax)
	logerr = np.log10(err)
	statout = stats.linregress(logh, logerr)
	slope = statout[0]; intercept = statout[1]
	fit = slope*logh + intercept
	
	fig = plt.figure(figsize = (7,5), facecolor = 'w', edgecolor = 'w')
	ax = fig.add_subplot(1,1,1)
	ax.scatter(logh, logerr, marker = 'o', color = 'red')
	ax.plot(logh, fit, color = 'black')
	
	ax.set_xlabel(r'$log(\frac{1}{h})$', fontsize = 'large')
	ax.set_ylabel(r'$log(\epsilon_{L_2})$', fontsize = 'large')
	
	print "L2 error order = ", slope
	
	# cleanup
	[os.remove('../testdata/testmesh%d.mat' % i) for i in range(N)]
	

#=============================================================================================================================================================
def testLinDiff():    
        # init mesh
        K0 = importInitMesh(matfile = initmeshfile)
        m = Mesh(K0)
        
        # set zero Neumann boundary conditions
        m.NeumannEdges = m.BoundaryEdges
        m.NumNeumannEdges = len(m.NeumannEdges)
        def gNeumann(Mesh = m): return [lambda p: 0.]
        
        # get boundary partition
        m.partitionNodes()
        
        # assemble stiffness and mass matrices
        a = Assemb(Mesh = m)
	a.AssembStiffMat()
	a.AssembMassMat()
        K = a.globalStiffMat
        M = a.globalMassMat
        
        # set source function
        def fsrc(Mesh = m):  return [lambda p,m,u : 0.]
                               
        # assemble LHS of final lin. alg problem
        SDC = 100. # Diffusion coefficient bumped up for quick diffusion
        def makeBlockMat(x): return SDC * x
        def getlhs(K,M): return M
        
        # define the PDE
        pde = Parabolic(Mesh = m, StiffMat = a.globalStiffMat, MassMat = a.globalMassMat)
        pde.setNeumannFunc = gNeumann
        pde.setSrcFunc = fsrc
        pde.AssembBlockStiffMat = makeBlockMat
        pde.AssembBlockMassMat = makeBlockMat
        pde.AssembLHS = getlhs
        import linsolv
        K_free = makeBlockMat(pde.getFreeNodeArray(K))
        
        # initial condition
        f0 = [lambda p: 1. if np.sqrt(p[0]**2 + p[1]**2) <=0.1 else 0.]
        u0 = np.zeros([m.NumNodes, 1])
        for i, node in enumerate(m.Nodes): u0[i,0] = f0[0](node)
        
        # start time loop 
        outfile_fmt = os.path.join(testdata_dir, 'testdiff%d.dat')
        np.savetxt(outfile_fmt % 0, u0)
        dt = 0.5e-3
        
        print "dt = ", dt 
        print "Mesh size = ", m.Diam
        print "Characteristic time step # = ", int((1./SDC)/dt)
        raw_input('Press any key to start time loop')
        
        NSteps = 100
        u_old = u0
        err = []
        loop = True
        if loop:
                for i in range(NSteps):
                        print 'Timestep: ', i
                        pde.u = u_old
                        M_free, vecs = pde.AssembPDE()
                        F = vecs[0]; G = vecs[1]; D = vecs[2]
                        x_old = pde.wrapSol(u_old)
                        rhs = linsolv.mul(a = M_free, b = x_old) - dt * linsolv.mul(a = K_free, b = x_old) + dt * F + \
                              dt * G - linsolv.mul(a = M_free, b = D)
                        x_new = pde.Solve(M_free, rhs)
                        u_new = pde.unwrapSol(x_new)
                        
                        # write to file and update
                        np.savetxt(outfile_fmt%(i+1), u_new)     
                        err.append(abs(x_new - x_old).max())
                        u_old = u_new
                  
        # pattern animations
        print '\nTime stepping ended...'
        raw_input('Press any key to start animation...')
        p = Plot(Mesh = m)
        fig = plt.figure(figsize = (7,5), facecolor = 'w', edgecolor = 'w')
        datafilelist = []
        [datafilelist.append(outfile_fmt % x) for x in range(NSteps)]
        ax = fig.add_subplot(1, 1, 1)
        p.ax = ax
        p.patternAnimate(dataFileList = datafilelist, Component = 0, delay = 0.01,
                         frameRate = 10, Prefix = '../testdata/testdiff')

        # plot convergence
        fig_err = plt.figure(figsize = (7,5), facecolor = 'w', edgecolor = 'w')
        ax_err = fig_err.add_subplot(1,1,1)
        ax_err.set_xlabel('Timestep', fontsize = 'xx-large'); ax_err.set_ylabel('Convergence to steady state', fontsize = 'xx-large')
        ax_err.plot(err)
#============================================================================================================================================================

if __name__ == '__main__':
	#testMesh()
	#testShapeFnPlot()
	#testPatternPlot()
	#testAssembly()
	#testLinSolver()
	#testPoisson()
	#testPoissonErrorScaling()
	testLinDiff()

plt.show()
